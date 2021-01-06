### Data
The data directory for a video is structured as follows:
```
video_name/
|-- rgb_256/
|   |-- 0001.png, etc.
|-- rgb_512/
|   |-- 0001.png, etc.
|-- mask/ (optional)
|-- |-- 01, etc.
|-- |-- |-- 0001.png, etc.   
|-- keypoints.json
|-- metadata.json
|-- homographies.txt (optional)
```
- `metadata.json` contains a dictionary:
```
'alphapose_input_size': [width, height]  # size of frames input to AlphaPose
'size_LR': [width, height]  # size of low-resolution frames (multiple of 16; height should be 256)
'n_textures': int  # number of texture maps required, calculated by 24*num_people + 1
'composite_order': [[1, 2, 3], [1, 3, 2], ... ]  # optional per-frame back-to-front layer compositing order
```
- `keypoints.json` is in the format output by the [AlphaPose Pose Tracker](https://github.com/MVIG-SJTU/AlphaPose).
See [here](https://github.com/MVIG-SJTU/AlphaPose/tree/master/trackers/PoseFlow) for details.