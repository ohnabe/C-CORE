import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import transforms
import utils
import pickle as pk

from logging import getLogger
logger = getLogger('__main__')


TRAIN_DF = 'train.pkl'
TEST_DF = 'test.pkl'


class CcoreDataset(torch.utils.data.Dataset):

    def __init__(self, df, transform, train=False):
        self.df = df
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.loc[i]
        data = row['band'] #'inc_angle'
        data = self.transform(data)

        if self.train:
            label = row['is_iceberg']
            return data, label
        else:
            return data, i


def get_dataset_as_pandas(file_path):

    df = pd.read_json(file_path)
    df['band_1'] = df['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
    df['band_2'] = df['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
    df['inc_angle'] = pd.to_numeric(df['inc_angle'], errors='coerce')
    df['inc_angle'] = df['inc_angle'].fillna(method='pad')
    df['band_3'] = (df['band_1'] + df['band_2']) / 2

    df['band'] =[np.dstack((x, y, z)).transpose(2, 0, 1) for x, y, z in zip(df['band_1'], df['band_2'], df['band_3'])]

    return df


def getCcoreDataset(file_path, transforms, train):

    if train:
        DATA_DF = TRAIN_DF
    else:
        DATA_DF = TEST_DF

    if not os.path.exists(DATA_DF):
        logger.info('> create annotation file {}'.format(DATA_DF))
        #logger.info('> load json file {}'.format(file_path))
        df = get_dataset_as_pandas(file_path)
        pk.dump(df, open(DATA_DF, 'wb'))
    else:
        df = pk.load(open(DATA_DF, 'rb'))

    dataset = CcoreDataset(df, transform=transforms, train=train)
    return dataset



#if __name__ == '__main__':
    #train = getCcoreDataset('./data/data/processed')








