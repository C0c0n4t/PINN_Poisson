#!/bin/bash

sudo apt update
sudo apt upgrade

sudo apt install nvidia-cuda-toolkit

nvcc --version

# maybe not working yet, only with pacakage from 
# https://developer.nvidia.com/cuda-12-6-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu
sudo apt install nvidia-cudnn

sudo apt install libcudnn9-cuda-12
sudo apt install libcudnn9-cuda-12
sudo apt install libcudnn9-samples
