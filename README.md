# Geometric Back-projection Network for Point Cloud Classification
This repository is for Geometric Back-projection Network (GBNet) introduced in the following paper:  
**Geometric Back-projection Network for Point Cloud Classification**  
[Shi Qiu](https://shiqiu0419.github.io/), [Saeed Anwar](https://saeed-anwar.github.io/),  [Nick Barnes](http://users.cecs.anu.edu.au/~nmb/)    
IEEE Transactions on Multimedia (TMM), 2021

## Paper and Citation
The paper can be downloaded from [arXiv](https://arxiv.org/abs/1911.12885) and [IEEE early access](https://ieeexplore.ieee.org/document/9410405).  
If you find our paper/code is useful, please cite:

        @article{qiu2021geometric,
            title={Geometric Back-projection Network for Point Cloud Classification},
            author={Qiu, Shi and Anwar, Saeed and Barnes, Nick},
            journal={IEEE Transactions on Multimedia},
            year={2021},
            doi={10.1109/TMM.2021.3074240}
        }

## Network Architecture
<p align="center">
  <img width="900" src="https://github.com/ShiQiu0419/GBNet/blob/master/gbnet.png">
</p>

## Updates
* **23/04/2021** Codes for both ```ModelNet40``` and ```ScanObjectNN``` are available now. 
* **27/04/2021** Update ```model.py``` by adding ```class ABEM_Module(nn.Module)```.
* **29/04/2021** Pre-trained model (OA: **80.50%**, mAcc: **77.31%**) on ScanObjectNN ~~is available at google drive.~~
* **30/04/2021** Update a pre-trained model (OA: **80.99%**, mAcc: **78.21%**) on ScanObjectNN via [google drive](https://drive.google.com/file/d/1Fh17b3kruQiGzdgMV8kdzyEUanKsy-PV/view?usp=sharing).
* To be continued.

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
* put the pre-trained model under ```./pretrained```
* then run:
```
python main.py --exp_name=gbnet_modelnet40_eval --model=gbnet --dataset=modelnet40 --eval=True --model_path=pretrained/gbnet_modelnet40.t7
```

## ScanObjectNN Experiment 
**Train the model:**
* download [ScanObjectNN](https://github.com/hkust-vgd/scanobjectnn/), and extract both ```training_objectdataset_augmentedrot_scale75.h5``` and ```test_objectdataset_augmentedrot_scale75.h5``` files to ```./data```
* then run (more settings can be modified in ```main.py```):
```
python main.py --exp_name=gbnet_scanobjectnn --model=gbnet --dataset=ScanObjectNN
``` 

**Test the pre-trained model:**
* put the pre-trained model under ```./pretrained```
* then run:
```
python main.py --exp_name=gbnet_scanobjectnn_eval --model=gbnet --dataset=ScanObjectNN --eval=True --model_path=pretrained/gbnet_scanobjectnn.t7
```

## Pre-trained Models
* Python 3.6, Pytorch 0.4.0, Cuda 9.1
* 2 Nvidia P100 GPUs
* using default training settings as in ```main.py```

| Model            | Dataset             |#Points             | Data<br />Augmentation | Loss | Performance<br />on Test Set            | Download<br />Link   |
|:----------------:|:-------------------:|:-------------------:|:----------:|:-----------------:|:-------------------------------------------------------------------------------:|:------:|
| GBNet | ModelNet40 | 1024 | random scaling<br />and translation | cross-entropy<br />with label smoothing                 | overall accuracy: xx.x%<br />average class accuracy: xx.x%                                          | coming soon |
| GBNet | ScanObjectNN | 1024 | random scaling<br />and translation | cross-entropy<br />with label smoothing                 | overall accuracy: **80.99%**<br />average class accuracy: **78.21%**                                           | [google drive](https://drive.google.com/file/d/1Fh17b3kruQiGzdgMV8kdzyEUanKsy-PV/view?usp=sharing) |

For more discussions regarding the factors that may affect point cloud classification,  
please refer to the following paper:  
*[Revisiting Point Cloud Classification with a Simple and Effective Baseline](https://openreview.net/pdf?id=XwATtbX3oCz)*

## Acknowledgement
The code is built on [DGCNN](https://github.com/WangYueFt/dgcnn/tree/master/pytorch). We thank the authors for sharing the codes.
