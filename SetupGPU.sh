#!/bin/bash

sudo apt update
sudo apt upgrade

sudo apt install nvidia-cuda-toolkit

nvcc --version

sudo apt install nvidia-cudnn

sudo apt install libcudnn9-cuda-12
sudo apt install libcudnn9-cuda-12
sudo apt install libcudnn9-samples
