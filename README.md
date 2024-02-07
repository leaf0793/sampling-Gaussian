# The Sampling-Gaussian for Stereo Matching

This repository contains the code (in PyTorch) for "[The Sampling-Gaussian for Stereo Matching]" paper submission for ICML



## Usage

### Dependencies

- [Python 3.7](https://www.python.org/downloads/)
- [PyTorch(1.6.0+)](http://pytorch.org)
- torchvision 0.5.0
- [KITTI Stereo](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
- [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)


### Train, Finetune and Evaluation
As an example, use the following command to train a PSMNet on Scene Flow

```
python train.py
```

