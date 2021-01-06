#!/bin/bash
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

GPUS=0,1
DATA_PATH=./datasets/trampoline
bash datasets/prepare_iuv.sh $DATA_PATH
python train.py \
  --name trampoline \
  --dataroot $DATA_PATH \
  --batch_size 16 \
  --batch_size_upsample 6 \
  --gpu_ids $GPUS
python test.py \
  --name trampoline \
  --dataroot $DATA_PATH \
  --do_upsampling