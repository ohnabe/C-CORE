import numpy as np
import os
import json

def band_stack(x, y, z):
    print(x.shape)
    print(y.shape)
    print(z.shape)
    return np.dstack((x, y, z)).transpose(2, 0, 1)



def set_devices(device_str):
    os.environ['CUDA_VISIBLE_DEVICES'] = device_str
    num_devices = len(device_str.split(','))
    return num_devices


class AverageMeter(object):

    def __init__(self, window_size=None):
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.window_size = window_size


    def reset(self):
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        if self.window_size and (self.count >= self.window_size):
            self.reset()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_api_key_from_file(json_path):
    try:
        with open(json_path, "r") as f:
            parsed = json.load(f)
    except IOError:
        contiune

    return parsed.get('api_key') if parsed else None


def get_api_key_from_env():
    key_str = os.environ.get('HYPERDASH_API_KEY')
    return key_str

