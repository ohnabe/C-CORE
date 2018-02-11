#!/bin/bash

PRJ_DIR=$(pwd)

#nvidia-docker run --rm -it -v $PRJ_DIR:/work -w /work -v /media/realnabe/8eab9d25-cbda-47c2-a8ff-4cd9c26f9c43/Kaggle:/data -u $(id -u):$(id -g) pytorch:20171106 python /work/script-00.py
#nvidia-docker run --rm -it --ipc=host -v $PRJ_DIR:/work -w /work -v /media/realnabe/2e227579-2458-493f-9f8a-4533b68d6b12:/data -u $(id -u):$(id -g) pytorch:20171106 python /work/trainer_resnet101.py
#nvidia-docker run --rm -it --ipc=host -v $PRJ_DIR:/work -w /work -v /media/realnabe/8eab9d25-cbda-47c2-a8ff-4cd9c26f9c43/Kaggle:/data -u $(id -u):$(id -g) pytorch:20171106 python /work/submit_csv.py

#nvidia-docker run --rm -it --ipc=host -v $PRJ_DIR:/work -w /work -v /media/realnabe/2e227579-2458-493f-9f8a-4533b68d6b12:/data -u $(id -u):$(id -g) pytorch:20171106 python /work/train_predict.py

#nvidia-docker run --rm -it --ipc=host -v $PRJ_DIR:/work -w /work -u $(id -u):$(id -g) pytorch:20171106 python /work/getdataset.py

nvidia-docker run -it --rm --ipc=host -v $PRJ_DIR:/work -v $PRJ_DIR/data/data/processed:/data -w /work \
-e HYPERDASH_API_KEY=$HYPERDASH_API_KEY \
pytorch-chainercv:20171226 \
python /work/train.py \
--model_file model/Net.py \
--model_name Net_V1 \
--batchsize 2 \
--training_epoch 40 \
--experiment_name TEST \
--initial_lr 0.01 \
--lr_momentum 0.9 \
--lr_decay_rate 0.5 \
--lr_decay_epoch 3 \
--weight_decay 0.0005 \
--train_json /data/train.json \
--train_mode \
--random_angle 15.0 \
--expand_ratio 1.2 \
--crop_size 70 70 \
--seed 0 \
--gpus '1','3' \
--output_class 2