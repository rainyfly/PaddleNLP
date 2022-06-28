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
"""Finetuning on classification tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import multiprocessing

import paddle.fluid as fluid
import numpy as np

from reader.classification_reader import ClassifyReader
from model.unimo_finetune import UNIMOConfig
from model.tokenization import GptBpeTokenizer
from finetune.myclassifier import create_model, evaluate, predict
from utils.optimization import optimization
from utils.utils import get_time
from utils.args import print_arguments
from utils.init import init_pretraining_params, init_checkpoint
from args.classification_args import parser

args = parser.parse_args()

def main(args):
    """main"""
    model_config = UNIMOConfig(args.unimo_config_path)
    model_config.print_config()

    gpu_id = 0
    place = fluid.CUDAPlace(gpu_id)
    tokenizer = GptBpeTokenizer(vocab_file=args.unimo_vocab_file,
                                encoder_json_file=args.encoder_json_file,
                                vocab_bpe_file=args.vocab_bpe_file,
                                do_lower_case=args.do_lower_case)

    data_reader = ClassifyReader(tokenizer, args)

    if not (args.do_train or args.do_val or args.do_val_hard \
            or args.do_test or args.do_test_hard or args.do_diagnostic):
        raise ValueError("For args `do_train`, `do_val`, `do_val_hard`, `do_test`," \
                " `do_test_hard` and `do_diagnostic`, at least one of them must be True.")

    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed


    if args.do_val or args.do_val_hard or args.do_test or args.do_test_hard \
            or args.do_pred or args.do_pred_hard or args.do_diagnostic:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, graph_vars = create_model(
                    args,
                    pyreader_name='test_reader',
                    config=model_config)

        test_prog = test_prog.clone(for_test=True)

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if args.do_val or args.do_val_hard or args.do_test or args.do_test_hard \
            or args.do_pred or args.do_pred_hard or args.do_diagnostic:
        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if"
                             "only doing validation or testing!")
        init_checkpoint(
            exe,
            args.init_checkpoint,
            main_program=startup_prog)

    

    test_exe = exe

    dev_ret_history = [] # (steps, key_eval, eval)
    dev_hard_ret_history = [] # (steps, key_eval, eval)
    test_ret_history = []  # (steps, key_eval, eval)
    test_hard_ret_history = []  # (steps, key_eval, eval)
    
    # final eval on test set
    test_pyreader.decorate_tensor_provider(
        data_reader.data_generator(
            args.test_set,
            batch_size=args.batch_size,
            epoch=1,
            dev_count=1,
            shuffle=False))
    print("Final test result:")
    outputs = predict(test_exe, test_prog, test_pyreader, graph_vars)
    print(outputs)

        

if __name__ == '__main__':
    print_arguments(args)
    main(args)
