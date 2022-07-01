#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""args for classification task"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from utils.args import ArgumentGroup


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("load_checkpoint", str, None, "Init checkpoint to resume training from.")
model_g.add_arg("init_pretraining_params", str, None,
                "Init pre-training params which preforms fine-tuning from. If the "
                "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
model_g.add_arg("checkpoints", str, "checkpoints", "Path to save checkpoints.")
model_g.add_arg("save_checkpoints", bool, True, "Whether to save checkpoints")
model_g.add_arg("weight_sharing", bool, True, "If set, share weights between word embedding and masked lm.")
model_g.add_arg("unimo_vocab_file", str, './model_files/dict/unimo_en.vocab.txt', "unimo vocab")
model_g.add_arg("encoder_json_file", str, './model_files/dict/unimo_en.encoder.json', 'bpt map')
model_g.add_arg("vocab_bpe_file", str, './model_files/dict/unimo_en.vocab.bpe', "vocab bpe")
model_g.add_arg("unimo_config_path", str, "./model_files/config/unimo_base_en.json",
                "The file to save unimo configuration.")
model_g.add_arg("output_dir", str, "checkpoints", "Path to save checkpoints.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch", int, 3, "Number of epoches for fine-tuning.")
train_g.add_arg("learning_rate", float, 5e-5, "Learning rate used to train with warmup.")
train_g.add_arg("lr_scheduler", str, "linear_warmup_decay",
                "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("weight_decay", float, 0.01, "Weight decay rate for L2 regularizer.")
train_g.add_arg("warmup_proportion", float, 0.1,
                "Proportion of training steps to perform linear learning rate warmup for.")
train_g.add_arg("save_steps", int, 10000, "The steps interval to save checkpoints.")
train_g.add_arg("validation_steps", int, 1000, "The steps interval to evaluate model performance.")
train_g.add_arg("nccl_comm_num", int, 1, "NCCL comm num.")
train_g.add_arg("hierarchical_allreduce_inter_nranks", int, 7, "Hierarchical allreduce inter ranks.")
train_g.add_arg("use_hierarchical_allreduce", bool, False, "Use hierarchical allreduce or not.")
train_g.add_arg("use_fp16", bool, False, "Whether to use fp16 mixed precision training.")
train_g.add_arg("use_dynamic_loss_scaling", bool, False, "Whether to use dynamic loss scaling.")
train_g.add_arg("init_loss_scaling", float, 1.0,
                "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")
train_g.add_arg("incr_every_n_steps", int, 100, "Increases loss scaling every n consecutive.")
train_g.add_arg("decr_every_n_nan_or_inf", int, 2,
                "Decreases loss scaling every n accumulated steps with nan or inf gradients.")
train_g.add_arg("incr_ratio", float, 2.0,
                "The multiplier to use when increasing the loss scaling.")
train_g.add_arg("decr_ratio", float, 0.8,
                "The less-than-one-multiplier to use when decreasing.")
train_g.add_arg("beta1", float, 0.9, "beta1 for adam")
train_g.add_arg("beta2", float, 0.98, "beta2 for adam.")
train_g.add_arg("epsilon", float, 1e-06, "epsilon for adam.")

log_g = ArgumentGroup(parser, "logging", "logging related.")
log_g.add_arg("skip_steps", int, 10, "The steps interval to print loss.")
log_g.add_arg("verbose", bool, False, "Whether to output verbose log.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("train_set", str, None, "Path to training data.")
data_g.add_arg("test_set", str, None, "Path to test data.")
data_g.add_arg("test_hard_set", str, None, "Path to test_hard data.")
data_g.add_arg("dev_set", str, None, "Path to validation data.")
data_g.add_arg("dev_hard_set", str, None, "Path to validation_hard data.")
data_g.add_arg("diagnostic_set", str, None, "Path to diagnostic data.")
data_g.add_arg("max_seq_len", int, 512, "Number of words of the longest seqence.")
data_g.add_arg("batch_size", int, 32, "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("in_tokens", bool, False,
               "If set, the batch size will be the maximum number of tokens in one batch. "
               "Otherwise, it will be the maximum number of examples in one batch.")
data_g.add_arg("do_lower_case", bool, True,
               "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
data_g.add_arg("random_seed", int, 0, "Random seed.")
data_g.add_arg("num_labels", int, 2, "label number")
data_g.add_arg("max_query_length", int, 64, "Max query length.")
data_g.add_arg("max_answer_length", int, 100, "Max answer length.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda", bool, True, "If set, use GPU for training.")
run_type_g.add_arg("is_distributed", bool, False, "If set, then start distributed training.")
run_type_g.add_arg("use_fast_executor", bool, False, "If set, use fast parallel executor (in experiment).")
run_type_g.add_arg("num_iteration_per_drop_scope", int, 10, "Iteration intervals to drop scope.")
run_type_g.add_arg("do_train", bool, False, "Whether to perform training.")
run_type_g.add_arg("do_val", bool, False, "Whether to perform evaluation on dev data set.")
run_type_g.add_arg("do_val_hard", bool, False, "Whether to perform evaluation on dev hard data set.")
run_type_g.add_arg("do_test", bool, False, "Whether to perform evaluation on test data set.")
run_type_g.add_arg("do_test_hard", bool, False, "Whether to perform evaluation on test hard data set.")
run_type_g.add_arg("do_pred", bool, False, "Whether to predict on test data set.")
run_type_g.add_arg("do_pred_hard", bool, False, "Whether to predict on test hard data set.")
run_type_g.add_arg("do_diagnostic", bool, False, "Whether to predict on diagnostic data set.")
run_type_g.add_arg("pred_save", str, "./output/predict/test", "Whether to predict on test data set.")
run_type_g.add_arg("use_multi_gpu_test", bool, False, "Whether to perform evaluation using multiple gpu cards")
run_type_g.add_arg("eval_mertrics", str, "simple_accuracy", "eval_mertrics")

# Config for 4D Parallelism
data_g = ArgumentGroup(parser, "parallel", "Parallel configuration")
parser.add_argument("--use_sharding", type=str2bool, nargs='?', const=False, help="Use sharding Parallelism to training.")
parser.add_argument("--sharding_degree", type=int, default=1, help="Sharding degree. Share the parameters to many cards.")
parser.add_argument("--dp_degree", type=int, default=1, help="Data Parallelism degree.")
parser.add_argument("--mp_degree", type=int, default=1, help="Model Parallelism degree. Spliting the linear layers to many cards.")
parser.add_argument("--pp_degree", type=int, default=1, help="Pipeline Parallelism degree. Spliting the the model layers to different parts.")
parser.add_argument("--use_recompute", type=str2bool, nargs='?', const=False, help="Using the recompute to save the memory.")

# Other config
parser.add_argument("--seed", type=int, default=1234, help="Random seed for initialization.")
parser.add_argument("--num_workers", type=int, default=2, help="Num of workers for DataLoader.")
parser.add_argument("--check_accuracy", type=str2bool, nargs='?', const=False, help="Check accuracy for training process.")
parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu", "xpu"], help="select cpu, gpu, xpu devices.")
parser.add_argument("--lr_decay_style", type=str, default="cosine", choices=["cosine", "none"], help="Learning rate decay style.")
parser.add_argument("--share_folder", type=str2bool, nargs='?', const=False, help="Use share folder for data dir and output dir on multi machine.")

 # Default training config
parser.add_argument("--grad_clip", default=0.0, type=float, help="Grad clip for the parameter.")
parser.add_argument("--max_lr", default=1e-5, type=float, help="The initial max learning rate for Adam.")
parser.add_argument("--min_lr", default=5e-5, type=float, help="The initial min learning rate for Adam.")
parser.add_argument("--warmup_rate", default=0.01, type=float, help="Linear warmup over warmup_steps for learing rate.")
parser.add_argument("--decay_steps", default=360000, type=int, help="The steps use to control the learing rate. If the step > decay_steps, will use the min_lr.")
parser.add_argument("--max_steps", default=500000, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")

# Adam optimizer config
parser.add_argument("--adam_beta1", default=0.9, type=float, help="The beta1 for Adam optimizer. The exponential decay rate for the 1st moment estimates.")
parser.add_argument("--adam_beta2", default=0.999, type=float, help="The bate2 for Adam optimizer. The exponential decay rate for the 2nd moment estimates.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

# AMP config
parser.add_argument("--use_amp", type=str2bool, nargs='?', const=False, help="Enable mixed precision training.")
parser.add_argument("--enable_addto", type=str2bool, nargs='?', const=True, default=True, help="Whether to enable the addto strategy for gradient accumulation or not. This is only used for AMP training.")
parser.add_argument("--scale_loss", type=float, default=32768, help="The value of scale_loss for fp16. This is only used for AMP training.")
parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="The hidden dropout prob.")
parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1, help="The attention probs dropout prob.")

# Training steps config
parser.add_argument("--num_train_epochs", default=1, type=int, help="Total number of training epochs to perform.", )
parser.add_argument("--checkpoint_steps", type=int, default=500, help="Save checkpoint every X updates steps to the model_last folder.")
parser.add_argument("--logging_freq", type=int, default=10, help="Log every X updates steps.")
parser.add_argument("--eval_freq", type=int, default=500, help="Evaluate for every X updates steps.")
parser.add_argument("--eval_iters", type=int, default=10, help="Evaluate the model use X steps data.")


