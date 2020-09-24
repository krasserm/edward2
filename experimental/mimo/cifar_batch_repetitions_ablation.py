# coding=utf-8
# Copyright 2020 The Edward2 Authors.
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

"""Ablation study for choice of batch_repetitions, each setting somewhat tuned.
"""


def get_sweep(hyper):
  """Returns hyperparameter sweep."""
  # Two experiments ran: one with adjust_for_learning_rate=True; another False.
  adjust_for_learning_rate = True
  # Sweep over batch repetitions, adjusting each setting so the models always
  # run with the same number of training iterations.
  batch_repetitions_list = [1, 2, 4, 6]
  train_epochs_list = [250 * 4, 250 * 2, 250, int(250 * 4/6)]
  lr_decay_epochs_list = [
      [str(80*4), str(160*4), str(180*4)],
      [str(80*2), str(160*2), str(180*2)],
      ['80', '160', '180'],
      [str(int(80*4/6)), str(int(160*4/6)), str(int(180*4/6))],
  ]
  if adjust_for_learning_rate:
    base_learning_rate_list = [0.1/4, 0.1/2, 0.1, 0.1*2]
  else:
    base_learning_rate_list = [0.1, 0.1, 0.1, 0.1]
  domain = []
  if adjust_for_learning_rate:
    i = 0
  for [batch_repetitions,
       train_epochs,
       lr_decay_epochs,
       base_learning_rate] in zip(batch_repetitions_list,
                                  train_epochs_list,
                                  lr_decay_epochs_list,
                                  base_learning_rate_list):
    if adjust_for_learning_rate and i >= 2:
      break
    subdomain = [
        # hyper.sweep('seed', hyper.discrete(range(3))),
        hyper.sweep('per_core_batch_size', hyper.discrete([64, 128])),
        hyper.sweep('ensemble_size', hyper.discrete([2, 4])),
        hyper.sweep('l2', hyper.discrete([1e-4, 3e-4, 5e-4])),
        hyper.fixed('batch_repetitions', batch_repetitions, length=1),
        hyper.fixed('train_epochs', train_epochs, length=1),
        hyper.fixed('base_learning_rate', base_learning_rate, length=1),
        hyper.fixed('lr_decay_epochs', lr_decay_epochs, length=1),
    ]
    domain += [hyper.product(subdomain)]
    i += 1

  sweep = hyper.chainit(domain)
  return sweep
