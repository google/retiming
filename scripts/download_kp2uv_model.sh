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

mkdir -p ./checkpoints/kp2uv
MODEL_FILE=./checkpoints/kp2uv/latest_net_Kp2uv.pth
URL=https://www.robots.ox.ac.uk/~erika/retiming/pretrained_models/kp2uv.pth
wget -N $URL -O $MODEL_FILE
