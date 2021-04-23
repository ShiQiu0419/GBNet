# Geometric Back-projection Network for Point Cloud Classification
This repository is for Geometric Back-projection Network (GBNet) introduced in the following paper

[Shi Qiu](https://shiqiu0419.github.io/), [Saeed Anwar](https://saeed-anwar.github.io/),  [Nick Barnes](http://users.cecs.anu.edu.au/~nmb/), "Geometric Back-projection Network for Point Cloud Classification"  
IEEE Transactions on Multimedia (TMM), 2021

## Paper
The paper can be downloaded from [arXiv](https://arxiv.org/abs/1911.12885) and [IEEE early access](https://ieeexplore.ieee.org/document/9410405).

## Network Architecture
<p align="center">
  <img width="900" src="https://github.com/ShiQiu0419/GBNet/blob/master/gbnet.png">
</p>

## Implementation Platforms
* Python 3.6
* Pytorch 0.4.0 with Cuda 9.1
* Higher Python/Pytorch/Cuda versions should also be compatible

## Experiments
**Synthetic Dataset: [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)**  
Train the model:  
* ```cd ./modelnet40```
* download and unzip the dataset to ```./modelnet40/data```
* ```python main.py``` (specific training settings/arguments can be modified in ```main.py```)

**Real-object Dataset: [ScanObjectNN](https://github.com/hkust-vgd/scanobjectnn/)**

## Citation

If you find our paper is useful, please cite:

        @article{qiu2021geometric,
            title={Geometric Back-projection Network for Point Cloud Classification},
            author={Qiu, Shi and Anwar, Saeed and Barnes, Nick},
            journal={IEEE Transactions on Multimedia},
            year={2021},
            doi={10.1109/TMM.2021.3074240}
        }
