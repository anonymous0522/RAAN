# RAANet: Range-Aware Attention Network for LiDAR-based 3D Object Detection with Auxiliary Density Level Estimation

<img src="https://github.com/anonymous0522/RAAN/blob/master/docs/motivation.PNG" width="100%" height="100%">

## Installation
### The code base of this work is forked from [CenterPoint](https://github.com/tianweiy/CenterPoint). The environment and dataset setups are inditity.
0. The CUDA and Pytorch version that is used for this work:
~~~
'CUDA==10.0',
'torch==1.1.0',
'CUDNN==7.5.0'
~~~
**Warning:** We tried CUDA11.0+Torch1.7.1 on RTX3090, the AP performance is significantly lower than the aforementioned environment setup. 

1. Installation
~~~
git clone https://github.com/anonymous0522/RAAN.git
cd RAAN
~~~
Then follow the setup of CenterPoint: [INSTALL](https://github.com/anonymous0522/RAAN/blob/master/docs/INSTALL.md)

2. Data Preperation

Currently, we train and evaluate our method on NuScenes dataset. 

Please setup the dataset by [NUSC](https://github.com/anonymous0522/RAAN/blob/master/docs/NUSC.md) from CenterPoint.

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

