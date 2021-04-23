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

## Sythetic Data Experiment 
**Train the model:**
* download [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) and unzip it to ```./modelnet40/data```
* ```cd ./modelnet40```
* ```python main.py --exp_name=gbnet_modelnet40 --model=gbnet --dataset=modelnet40```  
(other settings can be modified in ```main.py```)  

**Test the pre-trained model:**
* put the pre-trained model to ```./modelnet40/pretrained```
* ```python main.py --exp_name=gbnet_modelnet40_eval --model=gbnet --dataset=modelnet40 --eval=True --model_path=pretrained/gbnet_modelnet40.t7_model.t7```

## Real-world Data Experiment 
**Train the model:**
* download [ScanObjectNN](https://github.com/hkust-vgd/scanobjectnn/) (Real-object Dataset) and unzip it to ```./scanobjectnn/data```
* ```cd ./scanobjectnn```
* ```python main.py``` (specific training settings/arguments can be modified in ```main.py```)  

**Test the pre-trained model:**
* put the pre-trained model to ```./scanobjectnn/pretrained```
* ```python main.py --exp_name=gbnet_1024_eval --eval=True --model_path=pretrained//gbnet_scanobjectnn_model.t7```

## Pre-trained models will be released soon.

## Citation

If you find our paper is useful, please cite:

        @article{qiu2021geometric,
            title={Geometric Back-projection Network for Point Cloud Classification},
            author={Qiu, Shi and Anwar, Saeed and Barnes, Nick},
            journal={IEEE Transactions on Multimedia},
            year={2021},
            doi={10.1109/TMM.2021.3074240}
        }
