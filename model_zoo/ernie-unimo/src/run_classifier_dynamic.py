# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning on classification tasks."""
import argparse
import os
import sys
import random
import json
import time
import yaml
import shutil
import multiprocessing

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from paddle.io import DataLoader, Dataset
from visualdl import LogWriter
import paddle.fluid as fluid

from paddlenlp.transformers import ErnieModel, ErnieForPretraining, ErniePretrainingCriterion, ErnieTokenizer
from paddlenlp.transformers import CosineAnnealingWithWarmupDecay, LinearAnnealingWithWarmupDecay
from paddlenlp.utils.batch_sampler import DistributedBatchSampler
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.ops import Topology
from paddlenlp.utils.log import logger
from reader.classification_reader import ClassifyReader, pad_batch_records
from model.unimo_finetune import UNIMOConfig, UNIMOModel
from model.tokenization import GptBpeTokenizer
from finetune.classifier import UNIMOClassifier, evaluate, predict
from utils.optimization import optimization
from utils.utils import get_time
from utils.args import print_arguments
from utils.init import init_pretraining_params, init_checkpoint
from args.classification_args import parser



def create_pretrained_dataset(
    args,
    tokenizer,
    data_world_size,
    data_world_rank,
    places=None,
    data_holders=None,
    current_step=0,
):
    train_ds = ClassifyReader(args.train_set, tokenizer, args)
    dev_ds = ClassifyReader(args.dev_set, tokenizer, args)
    test_ds = ClassifyReader(args.test_set, tokenizer, args)

    def _collate_data(data, stack_fn=Stack()):
        print('collect', data)
        data = pad_batch_records(data, tokenizer)
        # print('after collect', len(data))
        # num_fields = len(data[0])
        # out = [None] * num_fields
        # 0. token_ids,
        # 1. segment_ids,
        # 2. position_ids,
        # 3. input_mask
        # 4. label_id,
        # 5. qid,
        # for i in range(6):
        #     out[i] = stack_fn([x[i] for x in data])
        out = [paddle.to_tensor(x) for x in data]
        print(out)
        return out

    def loader(dataset, consumed_samples=0):
        batch_sampler = DistributedBatchSampler(
            dataset,
            batch_size=args.batch_size,
            num_replicas=data_world_size,
            rank=data_world_rank,
            shuffle=False,
            drop_last=True,
            consumed_samples=consumed_samples)
        data_loader = paddle.io.DataLoader(dataset=dataset,
                                           batch_sampler=batch_sampler,
                                           num_workers=1,
                                           worker_init_fn=None,
                                           collate_fn=_collate_data,
                                           return_list=False)
        return data_loader

    train_dl = loader(train_ds, args.batch_size * 1)
    dev_dl = loader(
        dev_ds,
        args.batch_size * current_step)
    test_dl = loader(test_ds, 0)

    return train_dl, dev_dl, test_dl

if __name__ == "__main__":
    args = parser.parse_args()
    worker_index = paddle.distributed.get_rank()
    worker_num = paddle.distributed.get_world_size()
    if worker_num > 1:
        paddle.distributed.init_parallel_env()
    tokenizer = GptBpeTokenizer(vocab_file=args.unimo_vocab_file,
                                encoder_json_file=args.encoder_json_file,
                                vocab_bpe_file=args.vocab_bpe_file,
                                do_lower_case=args.do_lower_case)
    train_dl, valid_dl, test_dl = create_pretrained_dataset(args, tokenizer, worker_num, worker_index)
    count = 0
    for data in train_dl():
        if count == 1:
            break
        count += 1
        print(data[0].shape)

# def all_gather(v):
#     if dist.get_world_size() <= 1:
#         return v.item()
#     ret = []
#     dist.all_gather(ret, v)
#     concat = paddle.concat(ret, axis=0)
#     return concat.mean().item()


# @paddle.no_grad()
# def run_evaluate(data_loader,
#                  model,
#                  criterion,
#                  iter_steps,
#                  log_writer,
#                  global_step,
#                  args,
#                  task_name="valid"):
#     model.eval()
#     all_loss, all_lm_loss, all_sop_loss = [], [], []

#     loss_global = {
#         "loss": paddle.to_tensor(0.0),
#         "lm_loss": paddle.to_tensor(0.0),
#         "sop_loss": paddle.to_tensor(0.0),
#     }

#     local_time = time.time()

#     for eval_step, batch in enumerate(data_loader):
#         input_ids, segment_ids, input_mask, masked_lm_positions, \
#         masked_lm_labels, next_sentence_labels = batch

#         # Create the model for the gpt pretrain
#         prediction_scores, seq_relationship_score = model(
#             input_ids=input_ids,
#             token_type_ids=segment_ids,
#             position_ids=None,
#             attention_mask=input_mask,
#             masked_positions=masked_lm_positions)

#         lm_loss, sop_loss = criterion(prediction_scores, seq_relationship_score,
#                                       masked_lm_labels, next_sentence_labels)
#         loss = lm_loss + sop_loss

#         loss_global["loss"] += loss.detach()
#         loss_global["lm_loss"] += lm_loss.detach()
#         loss_global["sop_loss"] += sop_loss.detach()

#         if eval_step >= iter_steps - 1:
#             log_info_dict = dict()
#             for k, v in loss_global.items():
#                 log_info_dict[k] = all_gather(v) / iter_steps
#                 v.subtract_(v)
#             if dist.get_rank() == 0:
#                 log_info_dict[
#                     "samples_per_second"] = iter_steps * args.micro_batch_size / (
#                         time.time() - local_time)
#                 logger.info(
#                     "%s step %d, batch: %d, loss: %f, lm_loss: %.6f, sop_loss: %.6f, speed: %.0f seqs/s"
#                     %
#                     (task_name, global_step, iter_steps, log_info_dict["loss"],
#                      log_info_dict["lm_loss"], log_info_dict["sop_loss"],
#                      log_info_dict["samples_per_second"]))

#                 for k, v in log_info_dict.items():
#                     log_writer.add_scalar("%s/%s" % (task_name, k), v,
#                                           global_step)

#             break

#     model.train()


# def set_seed(args):
#     if args.device == "cpu":
#         idx = 0
#     else:
#         idx = paddle.distributed.get_rank()
#     random.seed(args.seed + idx)
#     np.random.seed(args.seed + idx)
#     paddle.seed(args.seed + idx)


# def default_logdir() -> str:
#     """
#     Same default
#     """
#     import socket
#     from datetime import datetime

#     current_time = datetime.now().strftime("%b%d_%H-%M-%S")
#     return os.path.join("runs", current_time + "_" + socket.gethostname())


# def do_train(args):
#     paddle.set_device(args.device)

#     worker_index = paddle.distributed.get_rank()
#     worker_num = paddle.distributed.get_world_size()
#     local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))

#     if worker_num > 1:
#         paddle.distributed.init_parallel_env()

#     if args.dp_degree * args.sharding_degree == 1:
#         args.dp_degree = worker_num
#         args.sharding_degree = 1

#     args_post_process(args, worker_num)

#     logger.info('{:20}:{}'.format("paddle commit id", paddle.version.commit))
#     for arg in vars(args):
#         logger.info('{:20}:{}'.format(arg, getattr(args, arg)))

#     strategy = fleet.DistributedStrategy()
#     strategy.hybrid_configs = {
#         "dp_degree": args.dp_degree,
#         "mp_degree": 1,
#         "pp_degree": 1,
#         "sharding_degree": 1
#     }

#     fleet.init(is_collective=True, strategy=strategy)
#     hcg = fleet.get_hybrid_communicate_group()

#     # Create the random seed for the worker
#     set_seed(args)

#     assert args.dp_degree * args.sharding_degree == worker_num, \
#         "The product of degree num should be equal to worker_num."

#     # Create log write,
#     log_writer = None
#     if worker_index == 0:
#         log_writer = LogWriter(os.path.join(args.output_dir, default_logdir()))

#     # Define the input data

    # tokenizer = GptBpeTokenizer(vocab_file=args.unimo_vocab_file,
    #                             encoder_json_file=args.encoder_json_file,
    #                             vocab_bpe_file=args.vocab_bpe_file,
    #                             do_lower_case=args.do_lower_case)

    

#     # load config in checkpoint
#     global_step = 0
#     consumed_samples = 0
#     checkpoint_dir = os.path.join(args.output_dir, "model_last")
#     if os.path.exists(checkpoint_dir):
#         if os.path.isfile(os.path.join(checkpoint_dir, "./config.yml")):
#             with open(os.path.join(checkpoint_dir, "./config.yml"), "r") as f:
#                 step_config = yaml.load(f, Loader=yaml.FullLoader)
#                 assert step_config[
#                     "global_batch_size"] == args.global_batch_size, "Please ensure checkpoint global batch size is the same. Folder: {}".format(
#                         checkpoint_dir)
#                 consumed_samples = step_config["consumed_samples"]
#                 global_step = step_config["global_step"]

#     if args.model_name_or_path in pretrained_models_list:
#         model_config = model_class.pretrained_init_configuration[
#             args.model_name_or_path]
#         model_config["hidden_dropout_prob"] = args.hidden_dropout_prob
#         model_config[
#             "attention_probs_dropout_prob"] = args.attention_probs_dropout_prob
#         model = model_class(base_class(**model_config))
#     else:
#         model = model_class.from_pretrained(
#             args.model_name_or_path,
#             hidden_dropout_prob=args.hidden_dropout_prob,
#             attention_probs_dropout_prob=args.attention_probs_dropout_prob)

#     criterion = criterion_class()

#     if worker_index == 0:
#         # log the model config and args
#         model_config_json = json.dumps(model.get_model_config(),
#                                        ensure_ascii=False,
#                                        indent=2)
#         log_writer.add_text("model_config", model_config_json)
#         args_dict = {"paddle commit id": str(paddle.version.commit)}
#         for arg in vars(args):
#             args_dict[arg] = str(getattr(args, arg))
#         log_writer.add_text("args", json.dumps(args_dict, indent=2))

#     # Create the learning_rate sheduler and optimizer
#     if args.decay_steps is None:
#         args.decay_steps = args.max_steps
#     assert args.warmup_rate <= 1.0 and args.warmup_rate >= 0.0, "warmup_rate should be in [0, 1]"
#     args.warmup_steps = args.warmup_rate * args.max_steps

#     lr_scheduler = LinearAnnealingWithWarmupDecay(args.max_lr,
#                                                   args.min_lr,
#                                                   warmup_step=args.warmup_steps,
#                                                   decay_step=args.decay_steps,
#                                                   last_epoch=global_step)

#     clip = None
#     if args.grad_clip > 0:
#         clip = paddle.fluid.clip.GradientClipByGlobalNorm(
#             clip_norm=args.grad_clip)

#     decay_param = [
#         p.name for n, p in model.named_parameters()
#         if not any(nd in n for nd in ["bias", "norm"])
#     ]
#     logger.info("Using paddle.optimizer.AdamW.")
#     optimizer = paddle.optimizer.AdamW(
#         learning_rate=lr_scheduler if lr_scheduler is not None else args.max_lr,
#         beta1=args.adam_beta1,
#         beta2=args.adam_beta2,
#         epsilon=args.adam_epsilon,
#         parameters=model.parameters(),
#         weight_decay=args.weight_decay,
#         grad_clip=clip,
#         apply_decay_param_fun=lambda x: x in decay_param,
#         multi_precision=args.use_amp)

#     if args.use_amp:
#         scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
#         scaler = fleet.distributed_scaler(scaler)
#         model = paddle.amp.decorate(models=model,
#                                     level='O2',
#                                     save_dtype='float32')

#     if paddle.distributed.get_world_size() > 1:
#         model = fleet.distributed_model(model)
#         optimizer = fleet.distributed_optimizer(optimizer)

#     tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

#     data_file = get_train_data_file(args)

#     train_data_loader, valid_data_loader, test_data_loader = create_pretrained_dataset(
#         args,
#         data_file,
#         tokenizer,
#         data_world_size=worker_num,
#         data_world_rank=worker_index,
#         max_seq_len=args.max_seq_len,
#         current_step=global_step)

#     # load checkpoint vars
#     if os.path.exists(checkpoint_dir):
#         if os.path.isfile(os.path.join(checkpoint_dir, "./config.yml")):
#             logger.info("Try to load checkpoint from %s " % checkpoint_dir)
#             opt_path = os.path.join(checkpoint_dir, "model_state.pdopt")
#             params_path = os.path.join(checkpoint_dir, "model_state.pdparams")

#             if os.path.exists(opt_path):
#                 opt_dict = paddle.load(opt_path)
#                 optimizer.set_state_dict(opt_dict)
#                 model_dict = paddle.load(params_path)
#                 model.set_state_dict(model_dict)
#             else:
#                 logger.warning("No optimizer checkpoint file found in %s." %
#                                opt_path)
#             logger.info(
#                 "Checkpoint loaded from global step: {}".format(global_step))

#     loss_global = {
#         "loss": paddle.to_tensor(0.0),
#         "lm_loss": paddle.to_tensor(0.0),
#         "sop_loss": paddle.to_tensor(0.0),
#     }
#     tic_train = time.time()
#     while True:
#         # If not call valid_data_loader, the enumerate will call valid_data_loader
#         # many times. and start a new random dataloader.
#         valid_data_loader = valid_data_loader()
#         test_data_loader = test_data_loader()

#         # time count
#         train_reader_cost = 0.0
#         train_run_cost = 0.0
#         reader_start = time.time()

#         for step, batch in enumerate(train_data_loader()):
#             train_reader_cost += time.time() - reader_start
#             train_start = time.time()

#             # 0. input_ids,
#             # 1. segment_ids,
#             # 2. input_mask,
#             # 3. masked_lm_positions,
#             # 4. masked_lm_labels,
#             # 5. next_sentence_labels

#             input_ids, segment_ids, input_mask, masked_lm_positions, \
#             masked_lm_labels, next_sentence_labels = batch

#             with paddle.amp.auto_cast(args.use_amp,
#                                       custom_black_list=[
#                                           "reduce_sum",
#                                           "c_softmax_with_cross_entropy",
#                                           "elementwise_div"
#                                       ],
#                                       level='O2'):

#                 # Create the model for the ernie pretrain
#                 prediction_scores, seq_relationship_score = model(
#                     input_ids=input_ids,
#                     token_type_ids=segment_ids,
#                     position_ids=None,
#                     attention_mask=input_mask,
#                     masked_positions=masked_lm_positions)

#                 lm_loss, sop_loss = criterion(prediction_scores,
#                                               seq_relationship_score,
#                                               masked_lm_labels,
#                                               next_sentence_labels)
#                 loss = lm_loss + sop_loss

#             if args.use_amp:
#                 scaler.scale(loss).backward()
#                 scaler.minimize(optimizer, loss)
#             else:
#                 loss.backward()
#                 optimizer.step()

#             optimizer.clear_grad()
#             train_run_cost += time.time() - train_start

#             # Skip for accumulate_steps in global step
#             if (step + 1) % args.accumulate_steps != 0:
#                 continue

#             global_step += 1

#             loss_global["loss"] += loss.detach()
#             loss_global["lm_loss"] += lm_loss.detach()
#             loss_global["sop_loss"] += sop_loss.detach()

#             if global_step % args.logging_freq == 0:
#                 log_info_dict = dict()
#                 log_info_dict["global_step"] = global_step
#                 for k, v in loss_global.items():
#                     log_info_dict[k] = all_gather(v) / args.logging_freq
#                     v.subtract_(v)
#                 if worker_index == 0:
#                     speed = args.logging_freq / (time.time() - tic_train)
#                     log_info_dict["learning_rate"] = lr_scheduler.get_lr()
#                     log_info_dict["steps_per_second"] = speed
#                     log_info_dict[
#                         "samples_per_second"] = speed * args.global_batch_size

#                     for k, v in log_info_dict.items():
#                         log_writer.add_scalar("train/%s" % k, v, global_step)

#                     common_loginfo = "global step %d, loss: %.9f, lm_loss: %.6f, sop_loss: %.6f, speed: %.2f steps/s, ips: %.2f seqs/s, learning rate: %.5e" % (
#                         global_step, log_info_dict["loss"],
#                         log_info_dict["lm_loss"], log_info_dict["sop_loss"],
#                         speed, log_info_dict["samples_per_second"],
#                         log_info_dict["learning_rate"])

#                     addition_info = ""
#                     if args.use_amp:
#                         amp_info = {
#                             "loss_scaling": scaler._scale.item(),
#                             "incr_count": scaler._incr_count,
#                             "decr_count": scaler._decr_count
#                         }
#                         addition_info = ", ".join("%s: %d" % (k, v)
#                                                   for k, v in amp_info.items())
#                         addition_info = " " + addition_info
#                         for k, v in amp_info.items():
#                             log_writer.add_scalar("amp/%s" % k, v, global_step)

#                     logger.info(common_loginfo + addition_info)

#                 tic_train = time.time()

#             if lr_scheduler is not None:
#                 lr_scheduler.step()

#             if global_step % args.eval_freq == 0:
#                 # TODO, check the input data of validation

#                 run_evaluate(valid_data_loader,
#                              model,
#                              criterion,
#                              args.eval_iters,
#                              log_writer,
#                              global_step,
#                              args,
#                              task_name="valid")
#                 tic_train = time.time()

#             def save_ckpt(output_dir, model, tokenizer, args, global_step):
#                 step_config = {
#                     "model_name": args.model_name_or_path,
#                     "global_step": global_step,
#                     "global_batch_size": args.global_batch_size,
#                     "consumed_samples": global_step * args.global_batch_size,
#                 }

#                 logger.debug("saving models to {}".format(output_dir))
#                 model_to_save = model._layers if isinstance(
#                     model, paddle.DataParallel) else model

#                 model_to_save.save_pretrained(output_dir)
#                 tokenizer.save_pretrained(output_dir)
#                 paddle.save(optimizer.state_dict(),
#                             os.path.join(output_dir, "model_state.pdopt"))

#                 with open(os.path.join(output_dir, "config.yml"), "w") as f:
#                     yaml.dump(step_config,
#                               f,
#                               encoding='utf-8',
#                               allow_unicode=True)

#             if global_step % args.save_steps == 0 or global_step >= args.max_steps:
#                 output_dir = os.path.join(args.output_dir,
#                                           "model_%d" % global_step)
#                 if worker_index == 0:
#                     save_ckpt(output_dir, model, tokenizer, args, global_step)

#                 if worker_num > 1:
#                     paddle.distributed.barrier()
#                 tic_train = time.time()

#             if global_step % args.checkpoint_steps == 0:
#                 output_dir = os.path.join(args.output_dir, "model_last")
#                 if worker_index == 0:
#                     if not os.path.exists(output_dir):
#                         os.mkdir(output_dir)
#                     output_dir_bak = os.path.join(args.output_dir,
#                                                   "model_last_bak")
#                     if os.path.exists(output_dir):
#                         if os.path.exists(output_dir_bak):
#                             shutil.rmtree(output_dir_bak)
#                         shutil.move(output_dir, output_dir_bak)
#                         os.mkdir(output_dir)
#                     save_ckpt(output_dir, model, tokenizer, args, global_step)

#                 if worker_num > 1:
#                     paddle.distributed.barrier()

#             if global_step >= args.max_steps:
#                 run_evaluate(test_data_loader,
#                              model,
#                              criterion,
#                              args.test_iters,
#                              log_writer,
#                              global_step,
#                              args,
#                              task_name="test")
#                 del train_data_loader
#                 return

# def main():
#     dev_ret_history = [] # (steps, key_eval, eval)
#     dev_hard_ret_history = [] # (steps, key_eval, eval)
#     test_ret_history = []  # (steps, key_eval, eval)
#     test_hard_ret_history = []  # (steps, key_eval, eval)
#     if args.do_train:
#         train_pyreader.start()
#         steps = 0
#         if warmup_steps > 0:
#             graph_vars["learning_rate"] = scheduled_lr

#         time_begin = time.time()
#         skip_steps = args.skip_steps
#         while True:
#             try:
#                 steps += 1
#                 if steps % skip_steps == 0:
#                     train_fetch_list = [
#                         graph_vars["loss"].name,
#                         graph_vars["accuracy"].name,
#                         graph_vars["num_seqs"].name
#                     ]
#                     if "learning_rate" in graph_vars:
#                         train_fetch_list.append(graph_vars["learning_rate"].name)
#                     res = train_exe.run(fetch_list=train_fetch_list)

#                     outputs = {"loss": np.mean(res[0])}
#                     if "learning_rate" in graph_vars:
#                         outputs["learning_rate"] = float(res[3][0])

#                     if args.verbose:
#                         verbose = "train pyreader queue size: %d, " % train_pyreader.queue.size(
#                         )
#                         verbose += "learning rate: %f" % (
#                             outputs["learning_rate"]
#                             if warmup_steps > 0 else args.learning_rate)
#                         print(verbose)

#                     current_example, current_epoch = data_reader.get_train_progress()
#                     time_end = time.time()
#                     used_time = time_end - time_begin
#                     print("%s - epoch: %d, progress: %d/%d, step: %d, ave loss: %f, speed: %f steps/s" % \
#                           (get_time(), current_epoch, current_example, num_train_examples, steps, \
#                           outputs["loss"], args.skip_steps / used_time))
#                     time_begin = time.time()
#                 else:
#                     train_exe.run(fetch_list=[])

#                 if nccl2_trainer_id == 0:
#                     if steps % args.save_steps == 0 and args.save_checkpoints:
#                         save_path = os.path.join(args.checkpoints,
#                                                  "step_" + str(steps))
#                         fluid.io.save_persistables(exe, save_path, train_program)

#                     if steps % args.validation_steps == 0:
#                         # evaluate dev set
#                         if args.do_val:
#                             test_pyreader.decorate_tensor_provider(
#                                 data_reader.data_generator(
#                                     args.dev_set,
#                                     batch_size=args.batch_size,
#                                     epoch=1,
#                                     dev_count=1,
#                                     shuffle=False))
#                             outputs = evaluate(args, test_exe, test_prog, test_pyreader, graph_vars, "dev")
#                             dev_ret_history.append((steps, outputs['key_eval'], outputs[outputs['key_eval']]))
                        
#                         # evaluate dev_hard set
#                         if args.do_val_hard:
#                             test_pyreader.decorate_tensor_provider(
#                                 data_reader.data_generator(
#                                     args.dev_hard_set,
#                                     batch_size=args.batch_size,
#                                     epoch=1,
#                                     dev_count=1,
#                                     shuffle=False))
#                             outputs = evaluate(args, test_exe, test_prog, test_pyreader, graph_vars, "dev_hard")
#                             dev_hard_ret_history.append((steps, outputs['key_eval'], outputs[outputs['key_eval']]))

#                         # evaluate test set
#                         if args.do_test:
#                             test_pyreader.decorate_tensor_provider(
#                                 data_reader.data_generator(
#                                     args.test_set,
#                                     batch_size=args.batch_size,
#                                     epoch=1,
#                                     dev_count=1,
#                                     shuffle=False))
#                             outputs = evaluate(args, test_exe, test_prog, test_pyreader, graph_vars, "test")
#                             test_ret_history.append((steps, outputs['key_eval'], outputs[outputs['key_eval']]))

#                         # evaluate test_hard set
#                         if args.do_test_hard:
#                             test_pyreader.decorate_tensor_provider(
#                                 data_reader.data_generator(
#                                     args.test_hard_set,
#                                     batch_size=args.batch_size,
#                                     epoch=1,
#                                     dev_count=1,
#                                     shuffle=False))
#                             outputs = evaluate(args, test_exe, test_prog, test_pyreader, graph_vars, "test_hard")
#                             test_hard_ret_history.append((steps, outputs['key_eval'], outputs[outputs['key_eval']]))

#                         # pred diagnostic set
#                         if args.do_diagnostic:
#                             test_pyreader.decorate_tensor_provider(
#                                 data_reader.data_generator(
#                                     args.diagnostic_set,
#                                     batch_size=args.batch_size,
#                                     epoch=1,
#                                     dev_count=1,
#                                     shuffle=False))
#                             qids, preds, probs = predict(test_exe, test_prog, test_pyreader, graph_vars, dev_count=1)
#                             save_path = args.pred_save + '.diagnostic.' + str(steps) + '.txt'
#                             print("testing {}, save to {}".format(args.diagnostic_set, save_path))
#                             with open(save_path, 'w') as f:
#                                 for id, s, p in zip(qids, preds, probs):
#                                     f.write('{}\t{}\t{}\n'.format(id, s, p))
                        
#                         # pred test set
#                         if args.do_pred:
#                             test_pyreader.decorate_tensor_provider(
#                                 data_reader.data_generator(
#                                     args.test_set,
#                                     batch_size=args.batch_size,
#                                     epoch=1,
#                                     dev_count=1,
#                                     shuffle=False))
#                             qids, preds, probs = predict(test_exe, test_prog, test_pyreader, graph_vars, dev_count=1)
#                             save_path = args.pred_save + '.test.' + str(steps) + '.txt'
#                             print("testing {}, save to {}".format(args.test_set, save_path))
#                             with open(save_path, 'w') as f:
#                                 for id, s, p in zip(qids, preds, probs):
#                                     f.write('{}\t{}\t{}\n'.format(id, s, p))

#                         # pred test hard set
#                         if args.do_pred_hard:
#                             test_pyreader.decorate_tensor_provider(
#                                 data_reader.data_generator(
#                                     args.test_hard_set,
#                                     batch_size=args.batch_size,
#                                     epoch=1,
#                                     dev_count=1,
#                                     shuffle=False))
#                             qids, preds, probs = predict(test_exe, test_prog, test_pyreader, graph_vars, dev_count=1)
#                             save_path = args.pred_save + '.test_hard.' + str(steps) + '.txt'
#                             print("testing {}, save to {}".format(args.test_hard_set, save_path))
#                             with open(save_path, 'w') as f:
#                                 for id, s, p in zip(qids, preds, probs):
#                                     f.write('{}\t{}\t{}\n'.format(id, s, p))

#             except fluid.core.EOFException:
#                 if args.save_checkpoints:
#                     save_path = os.path.join(args.checkpoints, "step_" + str(steps))
#                     fluid.io.save_persistables(exe, save_path, train_program)
#                 train_pyreader.reset()
#                 break

if __name__ == "__main__":
    args = parser.parse_args()
    do_train(args)
