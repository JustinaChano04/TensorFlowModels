# SlimYOLOv3: Narrower, Faster and Better for UAV Real-Time Applications

[![Paper](http://img.shields.io/badge/Paper-arXiv.1907.11093-B3181B?logo=arXiv)](https://arxiv.org/abs/1907.11093)

This subdirectory offers some tools for manipulating the structure of Keras
models. It is possible to use this folder to train SlimYOLOv3, however it
requires significant manual effort compared to regular training because of
repetitions of sparsity training, channel pruning, and fine tuning. As such,
it is not included in the main repo.

This subdirectory requires additional requirements that can be found in the
requirements.txt given here.

This folder was adapted from a term project for Purdue's ECE 570 class. The
final report can be found [term_paper.pdf](here). Some aspects such as the
config are yet to de adapted into this subdirectory in a format that is
compatible with TF Vision. The demo can be found [final.ipynb](here) and the
(poorly trained) checkpoints created in the term project can be found
[https://drive.google.com/drive/folders/14bWdAItB7IbPFprx--ds15y6O_IB4n40?usp=sharing](here).
The experiment was cut short due to resource constraints, so the paper's
experiment wasn't completely reproduced.

In order to use this subdirectory, you must first train YOLOv3 on Visdrone
from scratch.

Next, the model must be pruned, sparsity trained with L1 regularization on the
batch normalization layers, and then fine tuned without L1. This process is
repeated three times to give SlimYOLOv3.

Configurations and scripts to tie everything together are not compatible with
this version of the repo and will be updated later.
