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

import torch
from third_party.models.base_model import BaseModel
from . import networks


class Kp2uvModel(BaseModel):
    """This class implements the keypoint-to-UV model (inference only)."""
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='kpuv')
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- test options
        """
        BaseModel.__init__(self, opt)
        self.visual_names = ['keypoints', 'output_uv']
        self.model_names = ['Kp2uv']
        self.netKp2uv = networks.define_kp2uv(gpu_ids=self.gpu_ids)
        self.isTrain = False  # only test mode supported

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.keypoints = input['keypoints'].to(self.device)
        self.image_paths = input['path']

    def forward(self):
        """Run forward pass. This will be called by <test>."""
        output = self.netKp2uv.forward(self.keypoints)
        self.output_uv = self.output2rgb(output)

    def output2rgb(self, output):
        """Convert network outputs to RGB image."""
        pred_id, pred_uv = output
        _, pred_id_class = pred_id.max(1)
        pred_id_class = pred_id_class.unsqueeze(1)
        # extract UV from pred_uv (48 channels); select based on class ID
        selected_uv = -1 * torch.ones(pred_uv.shape[0], 2, pred_uv.shape[2], pred_uv.shape[3], device=pred_uv.device)
        for partid in range(1, 25):
            mask = (pred_id_class == partid).float()
            selected_uv *= (1. - mask)
            selected_uv += mask * pred_uv[:, (partid - 1) * 2:(partid - 1) * 2 + 2]
        pred_uv = selected_uv
        rgb = torch.cat([pred_id_class.float() * 10 / 255. * 2 - 1, pred_uv], 1)
        return rgb

    def optimize_parameters(self):
        pass
