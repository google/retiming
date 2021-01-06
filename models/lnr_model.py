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
import numpy as np
import torch.nn.functional as F


class LnrModel(BaseModel):
    """This class implements the layered neural rendering model for decomposing a video into layers."""
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='layered_video')
        parser.add_argument('--texture_res', type=int, default=16, help='texture resolution')
        parser.add_argument('--texture_channels', type=int, default=16, help='# channels for neural texture')
        parser.add_argument('--n_textures', type=int, default=25, help='# individual texture maps, 24 per person (1 per body part) + 1 for background')
        if is_train:
            parser.add_argument('--lambda_alpha_l1', type=float, default=0.01, help='alpha L1 sparsity loss weight')
            parser.add_argument('--lambda_alpha_l0', type=float, default=0.005, help='alpha L0 sparsity loss weight')
            parser.add_argument('--alpha_l1_rolloff_epoch', type=int, default=200, help='turn off L1 alpha sparsity loss weight after this epoch')
            parser.add_argument('--lambda_mask', type=float, default=50, help='layer matting loss weight')
            parser.add_argument('--mask_thresh', type=float, default=0.02, help='turn off masking loss when error falls below this value')
            parser.add_argument('--mask_loss_rolloff_epoch', type=int, default=-1, help='decrease masking loss after this epoch; if <0, use mask_thresh instead')
            parser.add_argument('--n_epochs_upsample', type=int, default=500,
                                help='number of epochs to train the upsampling module')
            parser.add_argument('--batch_size_upsample', type=int, default=16, help='batch size for upsampling')
            parser.add_argument('--jitter_rgb', type=float, default=0.2, help='amount of jitter to add to RGB')
            parser.add_argument('--jitter_epochs', type=int, default=400, help='number of epochs to jitter RGB')
        parser.add_argument('--do_upsampling', action='store_true', help='whether to use upsampling module')

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options
        """
        BaseModel.__init__(self, opt)
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['target_image', 'reconstruction', 'rgba_vis', 'alpha_vis', 'input_vis']
        self.model_names = ['LNR']
        self.netLNR = networks.define_LNR(opt.num_filters, opt.texture_channels, opt.texture_res, opt.n_textures, gpu_ids=self.gpu_ids)
        self.do_upsampling = opt.do_upsampling
        if self.isTrain:
            self.setup_train(opt)

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def setup_train(self, opt):
        """Setup the model for training mode."""
        print('setting up model')
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['total', 'recon', 'alpha_reg', 'mask']
        self.visual_names = ['target_image', 'reconstruction', 'rgba_vis', 'alpha_vis', 'input_vis']
        self.do_upsampling = opt.do_upsampling
        if not self.do_upsampling:
            self.visual_names += ['mask_vis']
        self.criterionLoss = torch.nn.L1Loss()
        self.criterionLossMask = networks.MaskLoss().to(self.device)
        self.lambda_mask = opt.lambda_mask
        self.lambda_alpha_l0 = opt.lambda_alpha_l0
        self.lambda_alpha_l1 = opt.lambda_alpha_l1
        self.mask_loss_rolloff_epoch = opt.mask_loss_rolloff_epoch
        self.jitter_rgb = opt.jitter_rgb
        self.do_upsampling = opt.do_upsampling
        self.optimizer = torch.optim.Adam(self.netLNR.parameters(), lr=opt.lr)
        self.optimizers = [self.optimizer]

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.target_image = input['image'].to(self.device)
        if self.isTrain and self.jitter_rgb > 0:
            # add brightness jitter to rgb
            self.target_image += self.jitter_rgb * torch.randn(self.target_image.shape[0], 1, 1, 1).to(self.device)
            self.target_image = torch.clamp(self.target_image, -1, 1)
        self.input_uv = input['uv_map'].to(self.device)
        self.input_id = input['pids'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.image_paths = input['image_path']

    def gen_crop_params(self, orig_h, orig_w, crop_size=256):
        """Generate random square cropping parameters."""
        starty = np.random.randint(orig_h - crop_size + 1)
        startx = np.random.randint(orig_w - crop_size + 1)
        endy = starty + crop_size
        endx = startx + crop_size
        return starty, endy, startx, endx

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        if self.do_upsampling:
            input_uv_up = F.interpolate(self.input_uv, scale_factor=2, mode='bilinear')
            crop_params = None
            if self.isTrain:
                # Take random crop to decrease memory requirement.
                crop_params = self.gen_crop_params(*input_uv_up.shape[-2:])
                starty, endy, startx, endx = crop_params
                self.target_image = self.target_image[:, :, starty:endy, startx:endx]
            outputs = self.netLNR.forward(self.input_uv, self.input_id, uv_map_upsampled=input_uv_up, crop_params=crop_params)
        else:
            outputs = self.netLNR(self.input_uv, self.input_id)
        self.reconstruction = outputs['reconstruction'][:, :3]
        self.alpha_composite = outputs['reconstruction'][:, 3]
        self.output_rgba = outputs['layers']
        n_layers = outputs['layers'].shape[1]
        layers = outputs['layers'].clone()
        layers[:, 0, -1] = 1  # Background layer's alpha is always 1
        layers = torch.cat([layers[:, l] for l in range(n_layers)], -2)
        self.alpha_vis = layers[:, 3:4]
        self.rgba_vis = layers
        self.mask_vis = torch.cat([self.mask[:, l:l+1] for l in range(n_layers)], -2)
        self.input_vis = torch.cat([self.input_uv[:, 2*l:2*l+2] for l in range(n_layers)], -2)
        self.input_vis = torch.cat([torch.zeros_like(self.input_vis[:, :1]), self.input_vis], 1)

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.loss_recon = self.criterionLoss(self.reconstruction[:, :3], self.target_image)
        self.loss_total = self.loss_recon
        if not self.do_upsampling:
            self.loss_alpha_reg = networks.cal_alpha_reg(self.alpha_composite * .5 + .5, self.lambda_alpha_l1, self.lambda_alpha_l0)
            alpha_layers = self.output_rgba[:, :, 3]
            self.loss_mask = self.lambda_mask * self.criterionLossMask(alpha_layers, self.mask)
            self.loss_total += self.loss_alpha_reg + self.loss_mask
        else:
            self.loss_mask = 0.
            self.loss_alph_reg = 0.
        self.loss_total.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def update_lambdas(self, epoch):
        """Update loss weights based on current epochs and losses."""
        if epoch == self.opt.alpha_l1_rolloff_epoch:
            self.lambda_alpha_l1 = 0
        if self.mask_loss_rolloff_epoch >= 0:
            if epoch == 2*self.mask_loss_rolloff_epoch:
                self.lambda_mask = 0
        elif epoch > self.opt.epoch_count:
            if self.loss_mask < self.opt.mask_thresh * self.opt.lambda_mask:
                self.mask_loss_rolloff_epoch = epoch
                self.lambda_mask *= .1
        if epoch == self.opt.jitter_epochs:
            self.jitter_rgb = 0

    def transfer_detail(self):
        """Transfer detail to layers."""
        residual = self.target_image - self.reconstruction
        transmission_comp = torch.zeros_like(self.target_image[:, 0:1])
        rgba_detail = self.output_rgba
        n_layers = self.output_rgba.shape[1]
        for i in range(n_layers - 1, 0, -1):  # Don't do detail transfer for background layer, due to ghosting effects.
            transmission_i = 1. - transmission_comp
            rgba_detail[:, i, :3] += transmission_i * residual
            alpha_i = self.output_rgba[:, i, 3:4] * .5 + .5
            transmission_comp = alpha_i + (1. - alpha_i) * transmission_comp
        self.rgba = torch.clamp(rgba_detail, -1, 1)

    def get_results(self):
        """Return results. This is different from get_current_visuals, which gets visuals for monitoring training.

        Returns a dictionary:
            original - - original frame
            recon - - reconstruction
            rgba_l* - - RGBA for each layer
            mask_l* - - mask for each layer
        """
        self.transfer_detail()
        # Split layers
        results = {'reconstruction': self.reconstruction, 'original': self.target_image}
        n_layers = self.rgba.shape[1]
        for i in range(n_layers):
            results[f'mask_l{i}'] = self.mask[:, i:i+1]
            results[f'rgba_l{i}'] = self.rgba[:, i]
            if i == 0:
                results[f'rgba_l{i}'][:, -1:] = 1.
        return results

    def freeze_basenet(self):
        """Freeze all parameters except for the upsampling module."""
        net = self.netLNR
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        self.set_requires_grad([net.encoder, net.decoder, net.final_rgba], False)
        net.texture.requires_grad = False