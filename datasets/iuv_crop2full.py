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

"""Convert UV crops to full UV maps."""
import os
import sys
import json
from PIL import Image
import numpy as np


def place_crop(crop, image, center_x, center_y):
    """Place the crop in the image at the specified location."""
    im_height, im_width = image.shape[:2]
    crop_height, crop_width = crop.shape[:2]

    left = center_x - crop_width // 2
    right = left + crop_width
    top = center_y - crop_height // 2
    bottom = top + crop_height

    adjusted_crop = crop  # remove regions of crop that go beyond image bounds
    if left < 0:
        adjusted_crop = adjusted_crop[:, -left:]
    if right > im_width:
        adjusted_crop = adjusted_crop[:, :(im_width - right)]
    if top < 0:
        adjusted_crop = adjusted_crop[-top:]
    if bottom > im_height:
        adjusted_crop = adjusted_crop[:(im_height - bottom)]
    crop_mask = (adjusted_crop > 0).astype(crop.dtype).sum(-1, keepdims=True)
    image[max(0, top):min(im_height, bottom), max(0, left):min(im_width, right)] *= (1 - crop_mask)
    image[max(0, top):min(im_height, bottom), max(0, left):min(im_width, right)] += adjusted_crop

    return image

def crop2full(keypoints_path, metadata_path, uvdir, outdir):
    """Create each frame's layer UVs from predicted UV crops"""
    with open(keypoints_path) as f:
        kp_data = json.load(f)

    # Get all people ids
    people_ids = set()
    for frame in kp_data:
        for skeleton in kp_data[frame]:
            people_ids.add(skeleton['idx'])
    people_ids = sorted(list(people_ids))

    with open(metadata_path) as f:
        metadata = json.load(f)

    orig_size = np.array(metadata['alphapose_input_size'][::-1])
    out_size = np.array(metadata['size_LR'][::-1])

    if 'people_layers' in metadata:
        people_layers = metadata['people_layers']
    else:
        people_layers = [[pid] for pid in people_ids]

    # Create output directories.
    for layer_i in range(1, 1 + len(people_layers)):
        os.makedirs(os.path.join(outdir, f'{layer_i:02d}'), exist_ok=True)
    print(f'Writing UVs to {outdir}')

    for frame in sorted(kp_data):
        for layer_i, layer in enumerate(people_layers, 1):
            out_path = os.path.join(outdir, f'{layer_i:02d}', frame)
            sys.stdout.flush()
            sys.stdout.write('processing frame %s\r' % out_path)
            uv_map = np.zeros([out_size[0], out_size[1], 4])
            for person_id in layer:
                matches = [p for p in kp_data[frame] if p['idx'] == person_id]
                if len(matches) == 0:  # person doesn't appear in this frame
                    continue
                skeleton = matches[0]
                kps = np.array(skeleton['keypoints']).reshape(17, 3)
                # Get kps bounding box.
                left = kps[:, 0].min()
                right = kps[:, 0].max()
                top = kps[:, 1].min()
                bottom = kps[:, 1].max()
                height = bottom - top
                width = right - left
                orig_crop_size = max(height, width)
                orig_center_x = (left + right) // 2
                orig_center_y = (top + bottom) // 2

                # read predicted uv map
                uv_crop_path = os.path.join(uvdir, f'{person_id:02d}_{os.path.basename(out_path)[:-4]}_output_uv.png')
                if os.path.exists(uv_crop_path):
                    uv_crop = np.array(Image.open(uv_crop_path))
                else:
                    uv_crop = np.zeros([256, 256, 3])

                # add person ID channel
                person_mask = (uv_crop[..., 0:1] > 0).astype('uint8')
                person_ids = (255 - person_id) * person_mask
                uv_crop = np.concatenate([uv_crop, person_ids], -1)

                # scale crop to desired output size
                # 256 is the crop size, 192 is the inner crop size
                out_crop_size = orig_crop_size * 256./192 * out_size / orig_size
                out_crop_size = out_crop_size.astype(np.int)
                uv_crop = uv_crop.astype(np.uint8)
                uv_crop = np.array(Image.fromarray(uv_crop).resize((out_crop_size[1], out_crop_size[0]), resample=Image.NEAREST))

                # scale center coordinate accordingly
                out_center_x = (orig_center_x * out_size[1] / orig_size[1]).astype(np.int)
                out_center_y = (orig_center_y * out_size[0] / orig_size[0]).astype(np.int)

                # Place UV crop in full UV map and save.
                uv_map = place_crop(uv_crop, uv_map, out_center_x, out_center_y)
            uv_map = Image.fromarray(uv_map.astype('uint8'))
            uv_map.save(out_path)


if __name__ == "__main__":
    import argparse
    arguments = argparse.ArgumentParser()
    arguments.add_argument('--dataroot', type=str)
    opt = arguments.parse_args()

    keypoints_path = os.path.join(opt.dataroot, 'keypoints.json')
    metadata_path = os.path.join(opt.dataroot, 'metadata.json')
    uvdir = os.path.join(opt.dataroot, 'kp2uv/test_latest/images')
    outdir = os.path.join(opt.dataroot, 'iuv')
    crop2full(keypoints_path, metadata_path, uvdir, outdir)
