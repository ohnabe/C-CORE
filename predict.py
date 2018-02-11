import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import Adam

import getdataset
import transforms
import argparse
from functools import partial
from importlib import import_module
import time
import shutil
import os
import re
import random
import utils
import json
import glob
from tqdm import tqdm
import pandas as pd

from logging import getLogger
logger = getLogger('__main__')

def parse_args():
    parser = argparse.ArgumentParser(description='Predictin Script for C-CORE')

    parser.add_argument('--model_file', type=str, default='models/ResNet50.py')
    parser.add_argument('--model_name', type=str, default='ResNet50')
    parser.add_argument('--trained_model', type=str, default='betstmodel.npz')
    parser.add_argument('--gpus', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--test_json', type=str, default='test.json')
    parser.add_argument('--crop_size', type=int, nargs='*', default=[224, 224])
    parser.add_argument('--output_class', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--test_mode', action='store_false')

    args = parser.parse_args()
    return args


def predict(model, test_loader):
    model.eval()

    output = []
    index = []

    for test_data, ind in test_loader:
        with torch.no_grad():
            test_data, ind = test_data.cuda(), ind.cuda()
            test_data_var, ind_var = Variable(test_data), Variable(ind)
            prob = F.softmax(model(test_data_var), dim=1)
            prob = prob.cpu().numpy()
            prob = prob[:, 1]
            output.extend(prob)
            index.extend(ind)
    df = pd.DataFrame(data={'is_iceberg':output, 'index':index})
    return df


def create_submit_csv(test_json, df):
    test_df = pd.read_json(test_json)
    df['id'] = test_df['id']
    df = df.drop(['index'], axis=1)
    return df[['id','is_iceberg']]


def main():

    args = parse_args()

    # Load training model
    if args.trained_model:
        if os.path.isfile(args.trained_model):
            # Set up Devices
            logger.info('> set gpu device {}'.format(args.gpus))
            num_cuda_devices = utils.set_devices(args.gpus)

            logger.info('> init model')
            ext = os.path.splitext(args.model_file)[1]
            model_path = '.'.join(os.path.split(args.model_file)).replace(ext, '')
            model = import_module(model_path)
            model = getattr(model, args.model_name)(args.output_class)
            if num_cuda_devices > 0:
                model.cuda()

            logger.info("=> loading checkpoint '{}'".format(args.trained_model))
            checkpoint = torch.load(args.trained_model)
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.trained_model))

        # Create Dataset
        logger.info('> Creating Test DataSet')
        test_transform = partial(
            transforms.transform_f, crop_size=args.crop_size, train=args.test_mode)
        test = getdataset.getCcoreDataset(args.test_json, test_transform, train=args.test_mode)

        logger.info('> create dataloader')
        test_loader = torch.utils.data.DataLoader(test, batch_size=args.batchsize, shuffle=False)

        logger.info('> Predict')
        #predict(model, test_loader)
        df = predict(model, test_loader)

        #create submit file
        logger.info('> Create submit file')
        df = create_submit_csv(args.test_json, df)
        dir = os.path.dirname(args.trained_model)
        out_path = os.path.join(dir, 'submit.csv')
        print(out_path)
        df.to_csv(out_path, index=False)

        logger.info('> end predict')



if __name__ == '__main__':
    import logging

    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    main()