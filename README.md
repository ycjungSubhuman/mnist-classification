# CSED703 Assignment 1 MNIST Classification Practice

## Environment

The code was developed and tested on a Linux environment, with following dependencies.

* Pytorch 1.0.1
* CUDA 10
* CuDNN 7

If this code does not work on your system, you can try running the code
in our Docker image. Install nvidia-docker and run `docker/runbash.sh`.

## Running test on pre-trained model

`python3 test.py`

This will print classificatino accuracy on MNIST test set with model 
loaded from the latest checkpoint in `checkpoint/` directory.

## Running training from scratch

`rm -rf checkpoints/ && python3 train.py`

