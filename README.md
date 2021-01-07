# Layered Neural Rendering in PyTorch

This repository contains training code for the examples in the SIGGRAPH Asia 2020 paper "[Layered Neural Rendering for Retiming People in Video](https://retiming.github.io/)."

<img src='./img/teaser.gif' height="160px"/>

This is not an officially supported Google product.


## Prerequisites
- Linux
- Python 3.6+
- NVIDIA GPU + CUDA CuDNN

## Installation
This code has been tested with PyTorch 1.4 and Python 3.8.

- Install [PyTorch](http://pytorch.org) 1.4 and other dependencies.
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.

## Data Processing
- Download the data for a video used in our paper (e.g. "reflection"):
```bash
bash ./datasets/download_data.sh reflection
```
- Or alternatively, download all the data by specifying `all`.
- Download the pretrained keypoint-to-UV model weights:
```bash
bash ./scripts/download_kp2uv_model.sh
``` 
The pretrained model will be saved at `./checkpoints/kp2uv/latest_net_Kp2uv.pth`.
- Generate the UV maps from the keypoints:
```bash
bash datasets/prepare_iuv.sh ./datasets/reflection
```
## Training
- To train a model on a video (e.g. "reflection"), run:
```bash
python train.py --name reflection --dataroot ./datasets/reflection --gpu_ids 0,1
```
- To view training results and loss plots, visit the URL http://localhost:8097.
Intermediate results are also at `./checkpoints/reflection/web/index.html`.

You can find more scripts in the `scripts` directory, e.g. `run_${VIDEO}.sh` which combines data processing, training, and saving layer results for a video. 

**Note**:
- It is recommended to use >=2 GPUs, each with >=16GB memory.
- The training script first trains the low-resolution model for `--num_epochs` at `--batch_size`, and then trains the upsampling module for `--num_epochs_upsample` at `--batch_size_upsample`.
If you do not need the upsampled result, pass `--num_epochs_upsample 0`.
- Training the upsampling module requires ~2.5x memory as the low-resolution model, so set `batch_size_upsample` accordingly.
The provided scripts set the batch sizes appropriately for 2 GPUs with 16GB memory.
- GPU memory scales linearly with the number of layers.

## Saving layer results from a trained model
- Run the trained model:
```bash
python test.py --name reflection --dataroot ./datasets/reflection --do_upsampling
```
- The results (RGBA layers, videos) will be saved to `./results/reflection/test_latest/`.
- Passing `--do_upsampling` uses the results of the upsampling module. If the upsampling module hasn't been trained (`num_epochs_upsample=0`), then remove this flag.

## Custom video
To train on your own video, you will have to preprocess the data:
1. Extract the frames, e.g.
    ```
    mkdir ./datasets/my_video && cd ./datasets/my_video 
    mkdir rgb && ffmpeg -i video.mp4 rgb/%04d.png
    ```
1. Resize the video to 256x448 and save the frames in `my_video/rgb_256`, and resize the video to 512x896 and save in `my_video/rgb_512`.
1. Run [AlphaPose and Pose Tracking](https://github.com/MVIG-SJTU/AlphaPose) on the frames. Save results as `my_video/keypoints.json`
1. Create `my_video/metadata.json` following [these instructions](docs/data.md).
1. If your video has camera motion, either (1) stabilize the video, or (2) maintain the camera motion by computing homographies and saving as `my_video/homographies.txt`.
See `scripts/run_cartwheel.sh` for a training example with camera motion, and see `./datasets/cartwheel/homographies.txt` for formatting.

**Note**: Videos that are suitable for our method have the following attributes:
- Static camera or limited camera motion that can be represented with a homography.
- Limited number of people, due to GPU memory limitations. We tested up to 7 people and 7 layers.
Multiple people can be grouped onto the same layer, though they cannot be individually retimed.
- People that move relative to the background (static people will be absorbed into the background layer).
- We tested a video length of up to 200 frames (~7 seconds).

## Citation
If you use this code for your research, please cite the following paper:
```
@inproceedings{lu2020,
  title={Layered Neural Rendering for Retiming People in Video},
  author={Lu, Erika and Cole, Forrester and Dekel, Tali and Xie, Weidi and Zisserman, Andrew and Salesin, David and Freeman, William T and Rubinstein, Michael},
  booktitle={SIGGRAPH Asia},
  year={2020}
}
```

## Acknowledgments
This code is based on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
