# Adaptive Heavy-ball Method

## The Role of Momentum Parameters in the Optimal Convergence of Adaptive Heavy-ball Methods
## Bibtex
@inproceedings{
tao2021the,
title={The Role of Momentum Parameters in the Optimal Convergence of Adaptive Polyak's Heavy-ball Methods},
author={Wei Tao and Sheng Long and Gaowei Wu and Qing Tao},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=L7WD8ZdscQ5
}


## Requirements
    Keras version >= 2.2.1
    Tensorflow version >= 1.13.0

We conduct experiments on a sever with 2 NVIDIA 2080Ti GPUs. 

## Deep Networks：
(1) 4-layer CNN: We design a simple 4-layer CNN architecture that consists two convolutional layers (32 filters of size 3 × 3), one max-pooling layer (2 × 2 window and 0.25 dropout) and one fully connected layer (128 hidden units and 0.5 dropout). We use weight decay and batch normalization to reduce over-fitting.

(2) Resnet-18.


## How to use
### Keras

Suppose the loss function is the categorical crossentropy,

``` python
from myopt import Ada_heavy_ball
