import numpy as np
import cv2 as cv
from chainercv import transforms
import pandas as pd
from skimage import io, transform




def cv_rotate(img, angle):
    img = img.transpose(1,2,0) / 255.0
    center = (img.shape[0] // 2, img.shape[1] // 2)
    r = cv.getRotationMatrix2D(center, angle, 1.0)
    img = cv.warpAffine(img, r, img.shape[:2])
    img = img.transpose(2, 0, 1) * 255.
    img = img.astype(np.float32)
    return img


def random_rotate(img, random_angle):
    if random_angle != 0:
        angle = np.random.uniform(-random_angle, random_angle)
        img = cv_rotate(img, angle)
    return img


def color_augmentation(img, pca_sigma):
    if pca_sigma != 0:
        img = transforms.pca_lighting(img, pca_sigma)
    return img


def standalization(img, mean, std):
    img -= mean[:, None, None]
    img /= std[:, None, None]
    return img


def random_flip(img):
    img = transforms.random_flip(img, x_random=True)
    return img


def random_expand(img, expand_ratio):
    if expand_ratio > 1:
        img = transforms.random_expand(img, max_ratio=expand_ratio)
    return img


def random_crop(img, crop_size):
    if tuple(crop_size) != (32, 32):
        img = transforms.random_crop(img, tuple(crop_size))
    return img


def resize(img, crop_size):
    img = transforms.resize(img, tuple(crop_size))
    return img


def transform_f(input, random_angle=15., expand_ratio=1.0, crop_size=(64, 64), train=True):
    img = input.copy()

    if train:
        img = random_rotate(img, random_angle)
        img = random_flip(img)
        img = random_expand(img, expand_ratio)
        img = random_crop(img, tuple(crop_size))
    else:
        img = random_rotate(img, random_angle)
        img = random_crop(img, tuple(crop_size))
    return img



