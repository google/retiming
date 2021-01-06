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

import cv2
from third_party.data.base_dataset import BaseDataset
from third_party.data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import torch
import numpy as np
import json


class LayeredVideoDataset(BaseDataset):
    """A dataset class for video layers.

    It assumes that the directory specified by 'dataroot' contains metadata.json, and the directories iuv, rgb_256, and rgb_512.
    The 'iuv' directory should contain directories named 01, 02, etc. for each layer, each containing per-frame UV images.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--height', type=int, default=256, help='image height')
        parser.add_argument('--width', type=int, default=448, help='image width')
        parser.add_argument('--trimap_width', type=int, default=20, help='trimap gray area width')
        parser.add_argument('--use_mask_images', action='store_true', default=False, help='use custom masks')
        parser.add_argument('--use_homographies', action='store_true', default=False, help='handle camera motion')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        rgbdir = os.path.join(opt.dataroot, 'rgb_256')
        if opt.do_upsampling:
            rgbdir = os.path.join(opt.dataroot, 'rgb_512')
        uvdir = os.path.join(opt.dataroot, 'iuv')
        self.image_paths = sorted(make_dataset(rgbdir, opt.max_dataset_size))
        n_images = len(self.image_paths)
        layers = sorted(os.listdir(uvdir))
        layers = [l for l in layers if l.isdigit()]
        self.iuv_paths = []
        for l in layers:
            layer_iuv_paths = sorted(make_dataset(os.path.join(uvdir, l), n_images))
            if len(layer_iuv_paths) != n_images:
                print(f'UNEQUAL NUMBER OF IMAGES AND IUVs: {len(layer_iuv_paths)} and {n_images}')
            self.iuv_paths.append(layer_iuv_paths)

        # set up per-frame compositing order
        with open(os.path.join(opt.dataroot, 'metadata.json')) as f:
            metadata = json.load(f)
        if 'composite_order' in metadata:
            self.composite_order = metadata['composite_order']
        else:
            self.composite_order = [tuple(range(1, 1 + len(layers)))] * n_images

        if opt.use_homographies:
            self.init_homographies(os.path.join(opt.dataroot, 'homographies.txt'), n_images)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains:
            image (tensor) - - the original RGB frame to reconstruct
            uv_map (tensor) - - the UV maps for all layers, concatenated channel-wise
            mask (tensor) - - the trimaps for all layers, concatenated channel-wise
            pids (tensor) - - the person IDs for all layers, concatenated channel-wise
            image_path (str) - - image path
        """
        # Read the target image.
        image_path = self.image_paths[index]
        target_image = self.load_and_process_image(image_path)

        # Read the layer IUVs and convert to network inputs.
        people_layers = [self.load_and_process_iuv(self.iuv_paths[l - 1][index], index) for l in
                         self.composite_order[index]]
        iuv_h, iuv_w = people_layers[0][0].shape[-2:]

        # Create the background layer UV from homographies.
        background_layer = self.get_background_inputs(index, iuv_w, iuv_h)

        uv_maps, masks, pids = zip(*([background_layer] + people_layers))
        uv_maps = torch.cat(uv_maps)  # [L*2, H, W]
        masks = torch.stack(masks)  # [L, H, W]
        pids = torch.stack(pids)  # [L, H, W]

        if self.opt.use_mask_images:
            for i in range(1, len(people_layers)):
                mask_path = os.path.join(self.opt.dataroot, 'mask', f'{i:02d}', os.path.basename(image_path))
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert('L').resize((masks.shape[-1], masks.shape[-2]))
                    mask = transforms.ToTensor()(mask) * 2 - 1
                    masks[i] = mask

        transform_params = self.get_params(do_jitter=self.opt.phase=='train')
        pids = self.apply_transform(pids, transform_params, 'nearest')
        masks = self.apply_transform(masks, transform_params, 'bilinear')
        uv_maps = self.apply_transform(uv_maps, transform_params, 'nearest')
        image_transform_params = transform_params
        if self.opt.do_upsampling:
            image_transform_params = { p: transform_params[p] * 2 for p in transform_params}
        target_image = self.apply_transform(target_image, image_transform_params, 'bilinear')

        return {'image': target_image, 'uv_map': uv_maps, 'mask': masks, 'pids': pids, 'image_path': image_path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)

    def get_params(self, do_jitter=False, jitter_rate=0.75):
        """Get transformation parameters."""
        if do_jitter:
            if np.random.uniform() > jitter_rate or self.opt.do_upsampling:
                scale = 1.
            else:
                scale = np.random.uniform(1, 1.25)
            jitter_size = (scale * np.array([self.opt.height, self.opt.width])).astype(np.int)
            start1 = np.random.randint(jitter_size[0] - self.opt.height + 1)
            start2 = np.random.randint(jitter_size[1] - self.opt.width + 1)
        else:
            jitter_size = np.array([self.opt.height, self.opt.width])
            start1 = 0
            start2 = 0
        crop_pos = np.array([start1, start2])
        crop_size = np.array([self.opt.height, self.opt.width])
        return {'jitter size': jitter_size, 'crop pos': crop_pos, 'crop size': crop_size}

    def apply_transform(self, data, params, interp_mode='bilinear'):
        """Apply the transform to the data tensor."""
        tensor_size = params['jitter size'].tolist()
        crop_pos = params['crop pos']
        crop_size = params['crop size']
        data = F.interpolate(data.unsqueeze(0), size=tensor_size, mode=interp_mode).squeeze(0)
        data = data[:, crop_pos[0]:crop_pos[0] + crop_size[0], crop_pos[1]:crop_pos[1] + crop_size[1]]
        return data

    def init_homographies(self, homography_path, n_images):
        """Read homography file and set up homography data."""
        with open(homography_path) as f:
            h_data = f.readlines()
        h_scale = h_data[0].rstrip().split(' ')
        self.h_scale_x = int(h_scale[1])
        self.h_scale_y = int(h_scale[2])
        h_bounds = h_data[1].rstrip().split(' ')
        self.h_bounds_x = [float(h_bounds[1]), float(h_bounds[2])]
        self.h_bounds_y = [float(h_bounds[3]), float(h_bounds[4])]
        homographies = h_data[2:2 + n_images]
        homographies = [torch.from_numpy(np.array(line.rstrip().split(' ')).astype(np.float32).reshape(3, 3)) for line
                        in
                        homographies]
        self.homographies = homographies

    def load_and_process_image(self, im_path):
        """Read image file and return as tensor in range [-1, 1]."""
        image = Image.open(im_path).convert('RGB')
        image = transforms.ToTensor()(image)
        image = 2 * image - 1
        return image

    def load_and_process_iuv(self, iuv_path, i):
        """Read IUV file and convert to network inputs."""
        iuv_map = Image.open(iuv_path).convert('RGBA')
        iuv_map = transforms.ToTensor()(iuv_map)
        uv_map, mask, pids = self.iuv2input(iuv_map, i)
        return uv_map, mask, pids

    def iuv2input(self, iuv, index):
        """Create network inputs from IUV.
        Parameters:
            iuv - - a tensor of shape [4, H, W], where the channels are: body part ID, U, V, person ID.
            index - - index of iuv

        Returns:
            uv (tensor) - - a UV map for a single layer, ready to pass to grid sampler (values in range [-1,1])
            mask (tensor) - - the corresponding mask
            person_id (tensor) - - the person IDs

        grid sampler indexes into texture map of size tile_width x tile_width*n_textures
        """
        # Extract body part and person IDs.
        part_id = (iuv[0] * 255 / 10).round()
        part_id[part_id > 24] = 24
        part_id_mask = (part_id > 0).float()
        person_id = (255 - 255 * iuv[-1]).round()  # person ID is saved as 255 - person_id
        person_id *= part_id_mask  # background id is 0
        maxId = self.opt.n_textures // 24
        person_id[person_id>maxId] = maxId

        # Convert body part ID to texture map ID.
        # Essentially, each of the 24 body parts for each person, plus the background have their own texture 'tile'
        # The tiles are concatenated horizontally to create the texture map.
        tex_id = part_id + part_id_mask * 24 * (person_id - 1)

        uv = iuv[1:3]
        # Convert the per-body-part UVs to UVs that correspond to the full texture map.
        uv[0] += tex_id

        # Get the mask.
        bg_mask = (tex_id == 0).float()
        mask = 1.0 - bg_mask
        mask = mask * 2 - 1  # make 1 the foreground and -1 the background mask
        mask = self.mask2trimap(mask)

        # Composite background UV behind person UV.
        h, w = iuv.shape[1:]
        bg_uv = self.get_background_uv(index, w, h)
        uv = bg_mask * bg_uv + (1 - bg_mask) * uv

        # Map to [-1, 1] range.
        uv[0] /= self.opt.n_textures
        uv = uv * 2 - 1
        uv = torch.clamp(uv, -1, 1)

        return uv, mask, person_id

    def get_background_inputs(self, index, w, h):
        """Return data for background layer at 'index'."""
        uv = self.get_background_uv(index, w, h)
        # normalize to correct range, of full texture atlas
        uv[0] /= self.opt.n_textures
        uv = uv * 2 - 1  # [0,1] -> [-1,1]
        uv = torch.clamp(uv, -1, 1)

        mask = -torch.ones(*uv.shape[1:])
        pids = torch.zeros(*uv.shape[1:])
        return uv, mask, pids

    def get_background_uv(self, index, w, h):
        """Return background layer UVs at 'index' (output range [0, 1])."""
        ramp_u = torch.linspace(0, 1, steps=w).unsqueeze(0).repeat(h, 1)
        ramp_v = torch.linspace(0, 1, steps=h).unsqueeze(-1).repeat(1, w)
        ramp = torch.stack([ramp_u, ramp_v], 0)
        if hasattr(self, 'homographies'):
            # scale to [0, orig width/height]
            ramp[0] *= self.h_scale_x
            ramp[1] *= self.h_scale_y
            # apply homography
            ramp = ramp.reshape(2, -1)  # [2, H, W]
            H = self.homographies[index]
            [xt, yt] = self.transform2h(ramp[0], ramp[1], torch.inverse(H))
            # scale from world to [0,1]
            xt -= self.h_bounds_x[0]
            xt /= (self.h_bounds_x[1] - self.h_bounds_x[0])
            yt -= self.h_bounds_y[0]
            yt /= (self.h_bounds_y[1] - self.h_bounds_y[0])
            # restore shape
            ramp = torch.stack([xt.reshape(h, w), yt.reshape(h, w)], 0)
        return ramp

    def transform2h(self, x, y, m):
        """Applies 2d homogeneous transformation."""
        A = torch.matmul(m, torch.stack([x, y, torch.ones(len(x))]))
        xt = A[0, :] / A[2, :]
        yt = A[1, :] / A[2, :]
        return xt, yt

    def mask2trimap(self, mask):
        """Convert binary mask to trimap with values in [-1, 0, 1]."""
        fg_mask = (mask > 0).float()
        bg_mask = (mask < 0).float()
        trimap_width = getattr(self.opt, 'trimap_width', 20)
        trimap_width *= bg_mask.shape[-1] / self.opt.width
        trimap_width = int(trimap_width)
        bg_mask = cv2.erode(bg_mask.numpy(), kernel=np.ones((trimap_width, trimap_width)), iterations=1)
        bg_mask = torch.from_numpy(bg_mask)
        mask = fg_mask - bg_mask
        return mask
