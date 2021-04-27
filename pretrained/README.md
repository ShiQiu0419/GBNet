# Pre-trained Models
Note: The following models were pre-produced using this repo, but not the reported ones in the paper.  
If you have interets in the original training logs, please feel free to contact us. 
* Python 3.6, Pytorch 0.4.0, Cuda 9.1
* 2 Nvidia P100 GPUs
* using default training settings as in ```main.py```

| Model            | Dataset             |#Points             | Data<br />Augmentation | Loss | Performance<br />on Test Set            | Download<br />Link   |
|:----------------:|:-------------------:|:-------------------:|:----------:|:-----------------:|:-------------------------------------------------------------------------------:|:------:|
| GBNet | ModelNet40 | 1024 | random scaling<br />and translation | cross-entropy<br />with label smoothing                 | overall accuracy: xx.x%<br />mean class accuracy: xx.x%                                          | coming soon |
| GBNet | ScanObjectNN | 1024 | random scaling<br />and translation | cross-entropy<br />with label smoothing                 | overall accuracy: xx.x%<br />mean class accuracy: xx.x%                                           | coming soon |

For more discussions regarding the factors that may affect point cloud classification,  
please refer to the following paper:  
*[Revisiting Point Cloud Classification with a Simple and Effective Baseline](https://openreview.net/pdf?id=XwATtbX3oCz)*
