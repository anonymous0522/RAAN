# RAANet: Range-Aware Attention Network for LiDAR-based 3D Object Detection with Auxiliary Density Level Estimation

## Anonymous submission

## Abstract

3D object detection from LiDAR data for autonomous driving has been making remarkable strides in recent years. Among the state-of-the-art methodologies, encoding point clouds into a bird's-eye view (BEV) has been demonstrated to be both effective and efficient. Different from perspective views, BEV preserves rich spatial and distance information between objects; and while farther objects of the same type do not appear smaller in the BEV, they contain sparser point cloud features. This fact weakens BEV feature extraction using shared-weight convolutional neural networks. In order to address this challenge, we propose Range-Aware Attention Network (RAANet), which extracts more
powerful BEV features and generates superior 3D object detections. The range-aware attention (RAA) convolutions significantly improve feature extraction for near as well as far objects. Moreover, we propose a novel auxiliary loss for density estimation to further enhance the detection accuracy of RAANet for occluded objects. It is worth to note that our proposed RAA convolution is lightweight and compatible to be integrated into any CNN architecture used for the BEV detection. Extensive experiments on the nuScenes dataset demonstrate that our proposed approach outperforms the state-of-the-art methods for LiDAR-based 3D object detection, with real-time inference speed of 16 Hz for the full version and 22 Hz for the lite version. The code is publicly available at an anonymous Github repository https://github.com/anonymous0522/RAAN.

<img src="https://github.com/anonymous0522/RAAN/blob/master/docs/motivation.PNG" width="100%" height="100%">

## Installation
### The code base of this work is forked from [CenterPoint](https://github.com/tianweiy/CenterPoint). The environment and dataset setups are inditity.
0. The CUDA and Pytorch version that is used for this work:
~~~
'CUDA==10.0',
'torch==1.1.0',
'CUDNN==7.5.0'
~~~

1. Installation
~~~
git clone https://github.com/anonymous0522/RAAN.git
cd RAAN
~~~
Then follow the setup of CenterPoint: [INSTALL](https://github.com/anonymous0522/RAAN/blob/master/docs/INSTALL.md)

2. Data Preperation
Currently, we train and evaluate our method on NuScenes dataset. Please setup the dataset by [NUSC](https://github.com/anonymous0522/RAAN/blob/master/docs/NUSC.md) from CenterPoint.

3. Examples of Training and Evaluation
Distributed Train:
~~~
python -m torch.distributed.launch —nproc_per_node=NUM_OF_GPU tools/train.py PATH_TO_CONFIG —work_dir PATH_TO_WORK_DIR
~~~
Normal Train:
~~~
python  tools/train.py PATH_TO_CONFIG —work_dir PATH_TO_WORK_DIR
~~~
Load and fine tune:
~~~
python3 tools/train.py PATH_TO_CONFIG --work_dir PATH_TO_WORK_DIR --load_from PATH_TO_MODEL
~~~
Test with test set:
~~~
python tools/dist_test.py PATH_TO_CONFIG —work_dir TPATH_TO_WORK_DIR --checkpoint PATH_TO_MODEL --testset —speed_test
~~~
With validation set:
~~~
python tools/dist_test.py PATH_TO_CONFIG —work_dir TPATH_TO_WORK_DIR --checkpoint PATH_TO_MODEL —speed_test
~~~
With distributed val:
~~~
python -m torch.distributed.launch —nproc_per_node=NUM_OF_GPU tools/dist_test.py PATH_TO_CONFIG —work_dir TPATH_TO_WORK_DIR --checkpoint PATH_TO_MODEL --testset —speed_test
~~~

## Main Results

### Object detection on VOC and COCO

<img src="https://github.com/anonymous0522/RAAN/blob/master/docs/result_table.PNG" width="100%" height="100%">
