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
import shutil
from hyperdash import Experiment
import sys


#from sklearn.metrics import accuracy_score


#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

DATA_DIR = './data'

from logging import getLogger
logger = getLogger('__main__')




def parse_args():
    parser = argparse.ArgumentParser(description='Training Script for C-CORE')

    parser.add_argument('--model_file', type=str, default='models/Net.py')
    parser.add_argument('--model_name', type=str, default='Net_V1')
    parser.add_argument('--gpus', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_json', type=str, default='train.json')
    parser.add_argument('--train_mode', action='store_true')
    parser.add_argument('--experiment_name', type=str, default='experiment_test')

    # Train settings
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--training_epoch', type=int, default=500)
    parser.add_argument('--initial_lr', type=float, default=0.05)
    parser.add_argument('--lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay_epoch', type=float, default=25)
    parser.add_argument('--lr_momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    # Data augmentation settings
    parser.add_argument('--random_angle', type=float, default=15.0)
    parser.add_argument('--pca_sigma', type=float, default=25.5)
    parser.add_argument('--expand_ratio', type=float, default=1.2)
    parser.add_argument('--crop_size', type=int, nargs='*', default=[224, 224])
    parser.add_argument('--output_class', type=int, default=100)

    args = parser.parse_args()
    return args


def create_result_dir(prefix):
    result_dir = 'results/{}_{}_0'.format(
        prefix, time.strftime('%Y-%m-%d_%H-%M-%S'))
    while os.path.exists(result_dir):
        i = result_dir.split('_')[-1]
        result_dir = re.sub('_[0-9]+$', result_dir, '_{}'.format(i))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    shutil.copy(__file__, os.path.join(result_dir, os.path.basename(__file__)))
    return result_dir


def training(train_loader, model, criterion, optimizer):

    model.train()

    running_loss = utils.AverageMeter()
    running_accuracy = utils.AverageMeter()

    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        data_var, target_var = Variable(data), Variable(target)
        output = model(data_var)
        loss = criterion(output, target_var)
        acc = accuracy(output.data, target, topk=(1,))
        running_loss.update(loss.data[0], data.size(0))
        running_accuracy.update(acc[0], data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    result = {'loss': running_loss.avg,
              'acc': running_accuracy.avg
                   }
    return result


def validate(val_loader, model, criterion):

    model.eval()

    val_loss_meter = utils.AverageMeter()
    val_acc_meter = utils.AverageMeter()

    for val_data, val_target in val_loader:
        val_data, val_target = val_data.cuda(), val_target.cuda()
        val_data_var, val_target_var = Variable(val_data), Variable(val_target)
        output = model(val_data_var)
        val_loss = criterion(output, val_target_var)
        val_acc = accuracy(output.data, val_target, topk=(1,))
        val_loss_meter.update(val_loss.data[0], val_data.size(0))
        val_acc_meter.update(val_acc[0], val_data.size(0))

    result = {'loss': val_loss_meter.avg,
              'acc': val_acc_meter.avg
              }
    return result


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    result= []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        result.append(correct_k.mul_(1.0 / batch_size)[0])
    return result


def save_checkpoint(state, is_best, result_dir, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(result_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(result_dir, filename),
                        os.path.join(result_dir, 'model_best.pth.tar'))



def main():

    args = parse_args()

    # set random seed
    #logger.info('> set random seed {}'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set up Devices
    #logger.info('> set gpu device {}'.format(args.gpus))
    num_cuda_devices = utils.set_devices(args.gpus)

    # Load model
    #logger.info('> load model {}'.format(args.model_name))
    ext = os.path.splitext(args.model_file)[1]
    model_path = '.'.join(os.path.split(args.model_file)).replace(ext, '')
    model = import_module(model_path)
    model = getattr(model, args.model_name)(args.output_class)
    if num_cuda_devices > 0:
        model = torch.nn.DataParallel(model)
        model.cuda()

    logger.info('> set optimizer')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.initial_lr, momentum=args.lr_momentum)

    # Create result dir
    result_dir = create_result_dir(args.model_name)

    fh_handler = logging.FileHandler(os.path.join(result_dir, "log"))
    fh_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(fh_handler)

    shutil.copy(args.model_file, os.path.join(
        result_dir, os.path.basename(args.model_file)))
    script_file_list = glob.glob('./*.py') + glob.glob('./*.sh')
    for file_name in script_file_list:
        shutil.copy(file_name, os.path.join(
            result_dir, os.path.basename(file_name)))
    with open(os.path.join(result_dir, 'args'), 'w') as fp:
        fp.write(json.dumps(vars(args)))
    print(json.dumps(vars(args), sort_keys=True, indent=4))

    # Create Dataset
    logger.info('> Creating DataSet')
    train_transform = partial(
        transforms.transform_f, random_angle=args.random_angle, expand_ratio=args.expand_ratio, crop_size=args.crop_size, train=True
    )
    train = getdataset.getCcoreDataset(args.train_json, train_transform, args.train_mode)

    val_transform = partial(
        transforms.transform_f, random_angle=args.random_angle, expand_ratio=args.expand_ratio, crop_size=args.crop_size, train=True
    )
    val = getdataset.getCcoreDataset(args.train_json, val_transform, args.train_mode)

    # Create DataLoader
    logger.info('> create dataloader')
    train_loader = torch.utils.data.DataLoader(train, batch_size= args.batchsize, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=4)

    # Training
    logger.info('> run training')
    best_prec = 0

    # Create Hyperdash Experiment
    logger.info('> Create Hyperdash Experiment {}'.format(args.experiment_name))
    exp = Experiment(args.experiment_name, api_key_getter=utils.get_api_key_from_env)

    for epoch in tqdm(range(args.training_epoch)):

        training_result = training(train_loader, model, criterion,
                                             optimizer
                                             )
        val_result = validate(val_loader, model, criterion)

        result_str = 'epoch : {} / {}\
        main/loss : {:.3f}\
        main/acc : {:.3f}\
        val/loss : {:.3f}\
        val/acc : {:.3f}'.format(
            epoch,
            args.training_epoch,
            training_result['loss'],
            training_result['acc'],
            val_result['loss'],
            val_result['acc']
        )
        logger.info(result_str)
        exp.log(result_str)

        prec1 = val_result['acc']

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec
        best_prec = max(prec1, best_prec)
        if is_best:
            save_checkpoint(state={
                'epoch': epoch + 1,
                #'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec': best_prec,
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, result_dir=result_dir)

        exp.metric('main/loss', training_result['loss'])
        exp.metric('val/loss', val_result['loss'])

    logger.info('> end training')
    exp.end()


if __name__ == '__main__':
    import logging

    st_handler = logging.StreamHandler(stream=sys.stdout)
    st_handler.setLevel(logging.INFO)
    logger.addHandler(st_handler)
    logger.setLevel(logging.INFO)
    main()
