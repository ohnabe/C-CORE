#!/bin/bash

#PRJ_DIR=$(pwd)

#git clone https://github.com/pytorch/pytorch.git
#nvidia-docker build -t pytorch:20171106 ./dockerfiles
nvidia-docker build -t pytorch-chainercv:20171226 .

#nvidia-docker run --rm -it -v $PRJ_DIR:/work -w /work -u $(id -u):$(id -g) pytorch:20171105 python3 /work/trainer_resnet101.py
