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
        data = pad_batch_records(data, tokenizer)
        # 0. token_ids,
        # 1. segment_ids,
        # 2. position_ids,
        # 3. input_mask
        # 4. label_id,
        # 5. qid,
        out = [paddle.to_tensor(x) for x in data]
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
                                           num_workers=2,
                                           worker_init_fn=None,
                                           collate_fn=_collate_data,
                                           return_list=False)
        return data_loader

    train_dl = loader(train_ds, 0)
    dev_dl = loader(dev_ds, 0)
    test_dl = loader(test_ds, 0)

    return train_dl, dev_dl, test_dl


@paddle.no_grad()
def run_evaluate(data_loader,
                 model,
                 global_step,
                 args,
                 task_name="valid"):
    model.eval()
    results = []
    labels = []
    for eval_step, batch in enumerate(data_loader):
        # 0. token_ids,
        # 1. segment_ids,
        # 2. position_ids,
        # 3. input_mask
        # 4. label_id,
        # 5. qid,

        input_ids, segment_ids, position_ids, masked_lm_positions, \
        label_ids, question_ids = batch

        emb_ids = {"word_embedding": input_ids, "sent_embedding": segment_ids, "pos_embedding": position_ids}

        # forward
        prediction_logits  = model(
            emb_ids, masked_lm_positions)
        
        loss, probs = paddle.nn.functional.softmax_with_cross_entropy(logits=prediction_logits, label=label_ids, return_softmax=True)
        results.append(probs)
        labels.append(label_ids)
    probs = paddle.concat(results, axis=0)
    label_ids = paddle.concat(labels, axis=0)
    result = paddle.metric.accuracy(input=probs, label=label_ids, k=1)
    logger.info("{} step {}, accuracy: {} ".format(task_name, global_step, result))
    model.train()
    return result


def set_seed(args):
    if args.device == "cpu":
        idx = 0
    else:
        idx = paddle.distributed.get_rank()
    random.seed(args.seed + idx)
    np.random.seed(args.seed + idx)
    paddle.seed(args.seed + idx)


def default_logdir() -> str:
    """
    Same default
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())


def do_train(args):
    paddle.set_device(args.device)

    worker_index = paddle.distributed.get_rank()
    worker_num = paddle.distributed.get_world_size()
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))

    if worker_num > 1:
        paddle.distributed.init_parallel_env()

    if args.dp_degree * args.sharding_degree == 1:
        args.dp_degree = worker_num
        args.sharding_degree = 1

    logger.info('{:20}:{}'.format("paddle commit id", paddle.version.commit))
    for arg in vars(args):
        logger.info('{:20}:{}'.format(arg, getattr(args, arg)))

    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": args.dp_degree,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": 1
    }

    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()

    # Create the random seed for the worker
    set_seed(args)

    assert args.dp_degree * args.sharding_degree == worker_num, \
        "The product of degree num should be equal to worker_num."

    # Create log write,
    log_writer = None
    if worker_index == 0:
        log_writer = LogWriter(os.path.join(args.output_dir, default_logdir()))

    # Define the input data

    tokenizer = GptBpeTokenizer(vocab_file=args.unimo_vocab_file,
                                encoder_json_file=args.encoder_json_file,
                                vocab_bpe_file=args.vocab_bpe_file,
                                do_lower_case=args.do_lower_case)
    
    train_data_loader, valid_data_loader, test_data_loader = create_pretrained_dataset(args, tokenizer, worker_num, worker_index)

    
    # Create model 
    model_config = UNIMOConfig(args.unimo_config_path)
    model_config.print_config()
    unimo_pretrain = UNIMOModel(config=model_config)
    model = UNIMOClassifier(unimo_pretrain, args.num_labels)
    

    # Create the learning_rate sheduler and optimizer
    global_step = 0
    if args.decay_steps is None:
        args.decay_steps = args.max_steps
    assert args.warmup_rate <= 1.0 and args.warmup_rate >= 0.0, "warmup_rate should be in [0, 1]"
    args.warmup_steps = args.warmup_rate * args.max_steps

    lr_scheduler = LinearAnnealingWithWarmupDecay(args.max_lr,
                                                  args.min_lr,
                                                  warmup_step=args.warmup_steps,
                                                  decay_step=args.decay_steps,
                                                  last_epoch=global_step)

    clip = None
    if args.grad_clip > 0:
        clip = paddle.fluid.clip.GradientClipByGlobalNorm(
            clip_norm=args.grad_clip)

    decay_param = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    logger.info("Using paddle.optimizer.AdamW.")
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler if lr_scheduler is not None else args.max_lr,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        grad_clip=clip,
        apply_decay_param_fun=lambda x: x in decay_param,
        multi_precision=args.use_amp)

    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
        scaler = fleet.distributed_scaler(scaler)
        model = paddle.amp.decorate(models=model,
                                    level='O2',
                                    save_dtype='float32')

    if paddle.distributed.get_world_size() > 1:
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)


    # load checkpoint vars
    from_checkpoint = False
    if os.path.exists(args.load_checkpoint):
        checkpoint_path = args.load_checkpoint
        from_checkpoint = True
    elif os.path.exists(args.init_pretraining_params):
        checkpoint_path = args.init_pretraining_params
        from_checkpoint = True
    if from_checkpoint:
        opt_path = os.path.join(checkpoint_path, "model_state.pdopt")
        params_path = os.path.join(checkpoint_path, "model_state.pdparams")
        if os.path.exists(opt_path):
            opt_dict = paddle.load(opt_path)
            optimizer.set_state_dict(opt_dict)
                
        else:
            logger.warning("No optimizer checkpoint file found in %s." %
                            opt_path)
        if os.path.exists(params_path):
            model_dict = paddle.load(params_path)
            model.set_state_dict(model_dict)
        else:
            logger.warning("No model checkpoint file found in %s." %
                            opt_path)
    


    tic_train = time.time()
    while True:
        # If not call valid_data_loader, the enumerate will call valid_data_loader
        # many times. and start a new random dataloader.

        # time count
        train_reader_cost = 0.0
        train_run_cost = 0.0
        reader_start = time.time()

        for step, batch in enumerate(train_data_loader()):
            train_reader_cost += time.time() - reader_start
            train_start = time.time()

            # 0. token_ids,
            # 1. segment_ids,
            # 2. position_ids,
            # 3. input_mask
            # 4. label_id,
            # 5. qid,

            input_ids, segment_ids, position_ids, masked_lm_positions, \
            label_ids, question_ids = batch

            emb_ids = {"word_embedding": input_ids, "sent_embedding": segment_ids, "pos_embedding": position_ids}

            # forward
            prediction_logits  = model(
                emb_ids, masked_lm_positions)
            loss, probs = paddle.nn.functional.softmax_with_cross_entropy(logits=prediction_logits, label=label_ids, return_softmax=True)
            loss = loss.mean()
            if worker_index == 0:
                log_writer.add_scalar(step=global_step, tag="loss", value=loss.numpy())
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            train_run_cost += time.time() - train_start

            global_step += 1

            if lr_scheduler is not None:
                lr_scheduler.step()

            if global_step % args.eval_freq == 0:
                # TODO, check the input data of validation

                accuracy = run_evaluate(valid_data_loader(),
                             model,
                             global_step,
                             args,
                             task_name="valid")
                if worker_index == 0:
                    log_writer.add_scalar(step=global_step, tag="accuracy", value=accuracy.numpy())
                tic_train = time.time()

            def save_ckpt(output_dir, model, global_step):
                logger.debug("saving models to {}".format(output_dir))
                model_to_save = model._layers if isinstance(
                    model, paddle.DataParallel) else model

                paddle.save(model.state_dict(),
                            os.path.join(output_dir, "model_state.pdparams"))
                paddle.save(optimizer.state_dict(),
                            os.path.join(output_dir, "model_state.pdopt"))


            if global_step % args.save_steps == 0 or global_step >= args.max_steps:
                output_dir = os.path.join(args.output_dir,
                                          "model_%d" % global_step)
                if worker_index == 0:
                    save_ckpt(output_dir, model,  global_step)

                if worker_num > 1:
                    paddle.distributed.barrier()
                tic_train = time.time()

            if global_step % args.checkpoint_steps == 0:
                output_dir = os.path.join(args.output_dir, "model_last")
                if worker_index == 0:
                    if not os.path.exists(output_dir):
                        os.mkdir(output_dir)
                    output_dir_bak = os.path.join(args.output_dir,
                                                  "model_last_bak")
                    if os.path.exists(output_dir):
                        if os.path.exists(output_dir_bak):
                            shutil.rmtree(output_dir_bak)
                        shutil.move(output_dir, output_dir_bak)
                        os.mkdir(output_dir)
                    save_ckpt(output_dir, model, global_step)

                if worker_num > 1:
                    paddle.distributed.barrier()

            if global_step >= args.max_steps:
                run_evaluate(test_data_loader(),
                             model,
                             global_step,
                             args,
                             task_name="test")
                del train_data_loader
                return
    

if __name__ == "__main__":
    args = parser.parse_args()
    do_train(args)
