# Geometric Back-projection Network for Point Cloud Classification
This repository is for Geometric Back-projection Network (GBNet) introduced in the following paper

[Shi Qiu](https://shiqiu0419.github.io/), [Saeed Anwar](https://saeed-anwar.github.io/),  [Nick Barnes](http://users.cecs.anu.edu.au/~nmb/), "Geometric Back-projection Network for Point Cloud Classification"  
IEEE Transactions on Multimedia (TMM), 2021

## Paper
The paper can be downloaded from [here (arXiv)](https://arxiv.org/abs/1911.12885)

## Introduction
As the basic task of point cloud learning, classification is fundamental but always challenging. To address some unsolved problems of existing methods, we propose a CNN based network leveraging an idea of error-correcting feedback structure to comprehensively capture the local features of 3D point clouds. Besides, we also enrich the explicit and implicit geometric information of point clouds in low-level 3D space and high-level feature space, respectively. By applying an attention module based on channel affinity, that focuses on distinct channels, the learned feature map of our network can effectively avoid redundancy. The performance on synthetic and real-world datasets demonstrate the superiority and applicability of our network. Comparing with other state-of-the-art methods, our approach balances accuracy and efficiency.

## Motivation
<p align="center">
  <img width="600" src="https://github.com/ShiQiu0419/Geometric-Feedback-Network-for-Point-Cloud-Classification/blob/master/overview2.png">
</p>

## Implementation
* Python 3.6
* Pytorch 0.4.0
* Cuda 9.1

## Experimental Results
**Synthetic Dataset: [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)**
<p align="center">
  <img width="600" src="https://github.com/ShiQiu0419/GFNet/blob/master/modelnet40.png">
</p>

**Real-object Dataset: [ScanObjectNN](https://github.com/hkust-vgd/scanobjectnn/)**
<p align="center">
  <img width="900" src="https://github.com/ShiQiu0419/GFNet/blob/master/scanobjectnn.png">
</p>

## Citation

If you find our paper is useful, please cite:

        @misc{qiu2019geometric,
            title={Geometric Back-projection Network for Point Cloud Classification},
            author={Shi Qiu and Saeed Anwar and Nick Barnes},
            year={2019},
            eprint={1911.12885},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
        }

## Codes
**The codes will be released Soon**
