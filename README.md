# RL Project 2020-1 (working on...)
Official Repository of CVPR 2019 Paper : RL-GAN-Net: A Reinforcement Learning Agent Controlled GAN Network for Real-Time Point Cloud Shape Completion

https://arxiv.org/abs/1904.12304

Requirements:

- `conda create -n <env_name> --file requirements_conda.txt python=3.6`
- `pip install -r requirements_pip.txt`
- `conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=XX.X -c pytorch` (XX.X: cuda version)
-  `mkdir data && ln -s <directory of train,test> shape_net_core_uniform_samples_2048_split`

Steps
* Visualize each training and testing step by using visdom.

1. Download data from https://github.com/optas/latent_3d_points.
2. Process Data with `Processdata2.m` to get complete point cloud (not incomplete!!)
3. Train the autoencoder using main.py and save the model
    - link data paths (train, test). see #TODO
    - open visdom server with port 8102 `python -m visdom.server -port 8102`
4. Generate GFV  using pretrained AE using GFVgen.py and store data
    - link pretrained model & train data path. see #TODO
5. Train GAN on the generated GFV data by by going into the GAN folder (trainer.py) and save model
6. Train RL by using pre-trained GAN and AE by running trainRL.py
    - First, process data with `Processdata.m` to get incomplete point cloud
    - link data paths (incomplete training dataset). see #TODO in `RL_params.py` 
7. Test with Incomplete data by running testRL.py
    - Not yet


TODOs
- valid set과 test set을 잘못 사용중.... 고쳐야함(main.py) 
- visdom 관련 코드 아예 없애든가 고치기

Credits:

1. https://github.com/optas/latent_3d_points
2. https://github.com/heykeetae/Self-Attention-GAN
3. https://github.com/lijx10/SO-Net (for chamfer distance)
4. https://github.com/sfujim/TD3



If you use this work for your projects, please take the time to cite our CVPR paper:

```
@InProceedings{Sarmad_2019_CVPR,
author = {Sarmad, Muhammad and Lee, Hyunjoo Jenny and Kim, Young Min},
title = {RL-GAN-Net: A Reinforcement Learning Agent Controlled GAN Network for Real-Time Point Cloud Shape Completion},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
