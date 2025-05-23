# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import argparse
import random

import paddle
import numpy as np
import cv2
import time

from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.utils import get_sys_env, utils
# from paddleseg.utils import  utils
from paddleseg.utils.logger import setup_logger
from paddleseg.core import train


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    # Common params
    parser.add_argument("--config", default='../configs/seaformer/seaformer_tiny_blueface.yml')
    parser.add_argument('--device',default='gpu',choices=['cpu', 'gpu'])
    parser.add_argument('--save_dir',default='./output')
    parser.add_argument("--precision",default="fp32",choices=["fp32", "fp16"],)
    parser.add_argument("--printlabels", default=['background', 'QPZZ', 'MDBD', 'MNYW', 'WW', 'LMPS', 'BMQQ', 'LMHH', 'KTAK'] )


    #以下是无需修改参数，内部已修改，外部无需修改---------------------------------------------------------------------
    parser.add_argument('--num_workers',default=0)
    parser.add_argument('--do_eval',default=True)
    parser.add_argument('--use_vdl',default=True)
    parser.add_argument('--use_ema',default=False)
    parser.add_argument('--iters', help='Iterations in training.', type=int)
    parser.add_argument('--batch_size',help='Mini batch size of one gpu or cpu. ',type=int)
    parser.add_argument('--learning_rate', help='Learning rate.', type=float)
    parser.add_argument('--save_interval',help='How many iters to save a model snapshot once during training.',type=int,default=50)
    parser.add_argument('--log_iters',help='Display logging information at every `log_iters`.',default=10,type=int)
    parser.add_argument('--seed',help='Set the random seed in training.',default=42,type=int)
    parser.add_argument(
        "--amp_level",
        default="O1",
        type=str,
        choices=["O1", "O2"],
        help=
        "Auto mixed precision level. Accepted values are “O1” and “O2”: O1 represent mixed precision, the input \
                data type of each operator will be casted by white_list and black_list; O2 represent Pure fp16, all operators \
                parameters and input data will be casted to fp16, except operators in black_list, don’t support fp16 kernel \
                and batchnorm. Default is O1(amp).")
    parser.add_argument(
        '--profiler_options',
        type=str,
        help='The option of train profiler. If profiler_options is not None, the train ' \
            'profiler is enabled. Refer to the paddleseg/utils/train_profiler.py for details.'
    )
    parser.add_argument(
        '--data_format',
        help=
        'Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".',
        type=str,
        default='NCHW')
    parser.add_argument(
        '--repeats',
        type=int,
        default=1,
        help=
        "Repeat the samples in the dataset for `repeats` times in each epoch.")
    parser.add_argument('--opts',
                        help='Update the key-value pairs of all options.',
                        nargs='+')

    # Runntime params
    parser.add_argument('--resume_model',
                        help='The path of the model to resume training.',
                        type=str)

    parser.add_argument('--keep_checkpoint_max',
                        help='Maximum number of checkpoints to save.',
                        type=int,
                        default=5)
    parser.add_argument('--early_stop_intervals',
                        help='Early Stop at args number of save intervals.',
                        type=int,
                        default=None)
    parser.add_argument(
        '--output_op',
        choices=['argmax', 'softmax', 'none'],
        default="argmax",
        help=
        "Select the op to be appended to the last of inference model, default: argmax."
        "In PaddleSeg, the output of trained model is logit (H*C*H*W). We can apply argmax and"
        "softmax op to the logit according the actual situation.")

    # Set multi-label mode
    parser.add_argument(
        '--use_multilabel',
        action='store_true',
        default=False,
        help='Whether to enable multilabel mode. Default: False.')
    parser.add_argument('--to_static_training',
                        action='store_true',
                        default=None,
                        help='Whether to enable to_static in training')
    parser.add_argument(
        "--input_shape",
        nargs='+',
        help=
        "Export the model with fixed input shape, e.g., `--input_shape 1 3 1024 1024`.",
        type=int,
        default=None)

    parser.add_argument('--for_fd',
                        action='store_true',
                        help="Export the model to FD-compatible format.")

    return parser.parse_args()


def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config'
    cfg = Config(args.config,
                 learning_rate=args.learning_rate,
                 iters=args.iters,
                 batch_size=args.batch_size,
                 to_static_training=args.to_static_training,
                 opts=args.opts)
    builder = SegBuilder(cfg)

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    utils.set_seed(args.seed)
    utils.set_device(args.device)
    utils.set_cv2_num_threads(args.num_workers)
    uniform_output_enabled = cfg.dic.get("uniform_output_enabled", False)
    if uniform_output_enabled:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        if os.path.exists(os.path.join(args.save_dir, "train_result.json")):
            os.remove(os.path.join(args.save_dir, "train_result.json"))
        with open(os.path.join(args.save_dir, "config.yaml"), "w") as f:
            yaml.dump(cfg.dic, f)
    print_mem_info = cfg.dic.pop('print_mem_info', True)
    shuffle = cfg.dic['train_dataset'].pop('shuffle', True)
    log_ranks = cfg.dic.pop('log_ranks', '0')

    if args.use_multilabel:
        if 'test_config' not in cfg.dic:
            cfg.dic['test_config'] = {'use_multilabel': True}
        else:
            cfg.dic['test_config']['use_multilabel'] = True

    # TODO refactor
    # Only support for the DeepLabv3+ model
    if args.data_format == 'NHWC':
        if cfg.dic['model']['type'] != 'DeepLabV3P':
            raise ValueError(
                'The "NHWC" data format only support the DeepLabV3P model!')
        cfg.dic['model']['data_format'] = args.data_format
        cfg.dic['model']['backbone']['data_format'] = args.data_format
        loss_len = len(cfg.dic['loss']['types'])
        for i in range(loss_len):
            cfg.dic['loss']['types'][i]['data_format'] = args.data_format

    model = utils.convert_sync_batchnorm(builder.model, args.device)
    train_dataset = builder.train_dataset
    # TODO refactor
    if args.repeats > 1:
        train_dataset.file_list *= args.repeats
    val_dataset = builder.val_dataset if args.do_eval else None
    # optimizer = builder.optimizer
    loss = builder.loss
    if not os.path.exists(os.path.join(args.save_dir,'res')): os.makedirs(os.path.join(args.save_dir,'res'))
    logger = setup_logger(log_ranks=log_ranks,log_file=os.path.join(args.save_dir,'res', '{}-{}.log'.format(str(args.config).split(os.path.sep)[-1].split('.')[0], time.strftime('%Y-%m-%d-%H-%M-%S'))))
    train(model,
          train_dataset,
          val_dataset=val_dataset,
          # optimizer=optimizer,
          lr_scheduler=cfg.dic['lr_scheduler'],
          builder=builder,
          save_dir=args.save_dir,
          iters=cfg.iters,
          max_epoch=cfg.dic['max_epoch'],
          batch_size=cfg.batch_size,
          early_stop_intervals=args.early_stop_intervals,
          resume_model=args.resume_model,
          save_interval=args.save_interval,
          log_iters=args.log_iters,
          num_workers=args.num_workers,
          use_vdl=args.use_vdl,
          use_ema=args.use_ema,
          losses=loss,
          keep_checkpoint_max=args.keep_checkpoint_max,
          test_config=cfg.test_config,
          precision=args.precision,
          amp_level=args.amp_level,
          profiler_options=args.profiler_options,
          to_static_training=cfg.to_static_training,
          logger=logger,
          print_mem_info=print_mem_info,
          shuffle=shuffle,
          uniform_output_enabled=uniform_output_enabled,
          cli_args=None if not uniform_output_enabled else args,
          printlabels=args.printlabels)


if __name__ == '__main__':
    args = parse_args()
    main(args)
