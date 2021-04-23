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

## ModelNet40 Experiment 
**Train the model:**
* download [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip), unzip and move ```modelnet40_ply_hdf5_2048``` folder to ```./data```
* then run (more settings can be modified in ```main.py```):    
```
python main.py --exp_name=gbnet_modelnet40 --model=gbnet --dataset=modelnet40
```   

**Test the pre-trained model:**
* put the pre-trained model to ```./pretrained```
* then run:
```
python main.py --exp_name=gbnet_modelnet40_eval --model=gbnet --dataset=modelnet40 --eval=True --model_path=pretrained/gbnet_modelnet40.t7_model.t7
```

## ScanObjectNN Experiment 
**Train the model:**
* download [ScanObjectNN](https://github.com/hkust-vgd/scanobjectnn/), and extract both ```training_objectdataset_augmentedrot_scale75.h5``` and ```test_objectdataset_augmentedrot_scale75.h5``` files to ```./data```
* then run (more settings can be modified in ```main.py```):
```
python main.py --exp_name=gbnet_scanobjectnn --model=gbnet --dataset=ScanObjectNN
``` 

**Test the pre-trained model:**
* put the pre-trained model to ```./pretrained```
* then run:
```
python main.py --exp_name=gbnet_scanobjectnn_eval --model=gbnet --dataset=ScanObjectNN --eval=True --model_path=pretrained/gbnet_scanobjectnn.t7
```

## Pre-trained Models
* Python 3.6, Pytorch 0.4.0, Cuda 9.1
* using default training settings as in ```main.py```

| Model            | Dataset             | Data Augmentation | Loss | Performance              | Link   |
|:----------------:|:-------------------:|:----------:|:-----------------:|:-------------------------------------------------------------------------------:|:------:|
| GBNet   | ModelNet40 | random scaling and translation | cross-entropy with label smoothing                 | Test set xxx% Overall Accuracy, xxx% Average Class Accuracy                                          | coming soon... |

## Citation

If you find our paper is useful, please cite:

        @article{qiu2021geometric,
            title={Geometric Back-projection Network for Point Cloud Classification},
            author={Qiu, Shi and Anwar, Saeed and Barnes, Nick},
            journal={IEEE Transactions on Multimedia},
            year={2021},
            doi={10.1109/TMM.2021.3074240}
        }
