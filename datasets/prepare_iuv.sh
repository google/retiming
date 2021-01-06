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

DATA_PATH=$1
# Predict UVs from keypoints and save the crops.
python run_kp2uv.py --model kp2uv --dataroot $DATA_PATH --results_dir $DATA_PATH
# Convert the cropped UVs to full UV maps.
python datasets/iuv_crop2full.py --dataroot $DATA_PATH
