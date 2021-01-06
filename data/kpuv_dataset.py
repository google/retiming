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

from third_party.data.base_dataset import BaseDataset
from PIL import Image, ImageDraw
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision.transforms as transforms


class KpuvDataset(BaseDataset):
    """A dataset class for keypoint data.

    It assumes that the directory specified by 'dataroot' contains the file 'keypoints.json'.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--inp_size', type=int, default=256, help='image size')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class by reading in keypoints.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.inp_size = opt.inp_size
        inner_crop_size = int(.75*self.inp_size)
        kps = []
        image_paths = []
        with open(os.path.join(self.root, 'keypoints.json'), 'rb') as f:
            kp_data = json.load(f)
        for frame in sorted(kp_data):
            for skeleton in kp_data[frame]:
                id = skeleton['idx']
                image_paths.append(f'{id:02d}_{frame}')
                kp = np.array(skeleton['keypoints']).reshape(17, 3)
                kp = self.crop_kps(kp, crop_size=self.inp_size, inner_crop_size=inner_crop_size)
                kps.append(kp)

        self.keypoints = kps
        self.image_paths = image_paths  # filenames for output UVs

        # for keypoint rendering
        self.cmap = plt.cm.get_cmap("hsv", 17)
        self.color_seq = np.array([ 9, 14,  6,  7, 13, 16,  2, 11,  3,  5, 10, 15,  1,  8,  0, 12,  4])
        self.pairs = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[11,12],[11,13],[13,15],[12,14],[14,16],[6,12],[5,11]]


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains keypoints and path
            keyoints (tensor) - - an RGB image representing a skeleton
            path (str) - - an identifying filename that can be used for saving the result
        """
        uv_path = self.image_paths[index]  # output UV path
        kps = self.keypoints[index]

        # draw keypoints
        kp_im = Image.new(size=(self.inp_size, self.inp_size), mode='RGB')
        draw = ImageDraw.Draw(kp_im)
        self.render_kps(kps, draw)
        kp_im = transforms.ToTensor()(kp_im)
        kp_im = 2 * kp_im - 1

        return {'keypoints': kp_im, 'path': uv_path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)

    def crop_kps(self, kps, crop_size=256, inner_crop_size=192):
        """'Crops' keypoints to fit into ['crop_size', 'crop_size'].

        Parameters:
            kps - - a numpy array of shape [17, 2], where the keypoint order is X, Y
            crop_size - - the new size of the world, which the keypoints will be centered inside
            inner_crop_size - - the box size that the keypoints will fit inside (must be <=crop_size)

        Returns keypoints mapped to fit inside a box of 'inner_crop_size', centered in 'crop_size'
        """
        # get coordinates of bounding box, in original image coordinates
        left = kps[:, 0].min()
        right = kps[:, 0].max()
        top = kps[:, 1].min()
        bottom = kps[:, 1].max()

        # map keypoints
        keypoints = kps.copy()
        center = ((right + left) // 2, (bottom + top) // 2)
        # first place center of bounding box at origin
        keypoints[:, 0] -= center[0]
        keypoints[:, 1] -= center[1]
        # scale bounding box to inner_crop_size
        scale = float(inner_crop_size) / max(right - left, bottom - top)
        keypoints[:, :2] *= scale
        # move center to crop_size//2
        keypoints[:, :2] += crop_size // 2
        new_kps = keypoints

        return new_kps

    def render_kps(self, keypoints, draw, thresh=1., min_weight=0.25):
        """Render skeleton as RGB image.

        Parameters:
            keypoints - - a numpy array of shape [17, 3], where the keypoint order is X, Y, score
            draw - - an ImageDraw object, which the keypoints will be drawn onto
            thresh - - keypoints with a confidence score below this value will have a color weighted by the score
            min_weight - - minimum weighting for color (scores will be mapped to the range [min_weight, 1])
        """
        # first draw keypoints
        ksize = 3
        for i in range(keypoints.shape[0]):
            x1 = keypoints[i,0] - ksize
            x2 = keypoints[i,0] + ksize
            y1 = keypoints[i,1] - ksize
            y2 = keypoints[i,1] + ksize
            if x1 < 0 or y1 < 0 or x2 > self.inp_size or y2 > self.inp_size:
                continue
            color = np.array(self.cmap(self.color_seq[i]))
            if keypoints.shape[1] > 2:
                score = keypoints[i,2]
                if score < thresh:  # weight color by confidence score
                    # first map [0,1] -> [min_weight, 1]
                    alpha_weight = score * (1.-min_weight) + min_weight
                    color[:3] *= alpha_weight
            color = (255*color).astype('uint8')
            draw.rectangle([x1, y1, x2, y2], fill=tuple(color))
        # now draw segments
        for pair in self.pairs:
            x1 = keypoints[pair[0],0]
            y1 = keypoints[pair[0],1]
            x2 = keypoints[pair[1],0]
            y2 = keypoints[pair[1],1]
            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or x1 > self.inp_size or y1 > self.inp_size or x2 > self.inp_size or y2 > self.inp_size:
                continue
            avg_color = .5*(np.array(self.cmap(self.color_seq[pair[0]])) + np.array(self.cmap(self.color_seq[pair[1]])))
            if keypoints.shape[1] > 2:
                score = min(keypoints[pair[0],2], keypoints[pair[1],2])
                if score < thresh:
                    alpha_weight = score * (1.-min_weight) + min_weight
                    avg_color[:3] *= alpha_weight # alpha channel weigh by score
            avg_color = (255*avg_color).astype('uint8')
            draw.line([x1,y1,x2,y2], fill=tuple(avg_color), width=3)