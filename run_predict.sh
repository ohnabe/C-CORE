#!/usr/bin/env bash

PRJ_DIR=$(pwd)


nvidia-docker run --rm -it --ipc=host -v $PRJ_DIR:/work -v $PRJ_DIR/data/data/processed:/data -w /work -u $(id -u):$(id -g) pytorch-chainercv:20171226 \
python /work/predict.py \
--model_file model/Net.py \
--model_name Net_V1 \
--trained_model /work/results/Net_V1_2018-01-03_15-00-47_0/model_best.pth.tar \
--test_json /data/test.json \
--crop_size 70 70 \
--seed 0 \
--gpus '3' \
--output_class 2 \
--batchsize 2 \
--test_mode
