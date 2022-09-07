from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms, utils
import torchvision.transforms.functional as TF

from tqdm import tqdm
import numpy as np
import json
import pandas as pd
import pickle
import matplotlib.pyplot as plt
# import skimage
# import skimage.io
# import skimage.transform
from PIL import Image
import time
import os
from os.path import join, exists
import copy
import random
from collections import OrderedDict
from sklearn.metrics import r2_score

from torch.nn import functional as F
from torchvision.models import Inception3, resnet18, resnet34, resnet50

from utils.image_dataset import *
from LR_models.siamese_model_rgb import *

"""
This script is for generating the prediction scores of blur detection model for images in 
sequences. A sequence of images are stored in a folder. An image is named by the year of 
the image plus an auxiliary index. E.g., '2007_0.png', '2007_1.png', '2008_0.png'.
"""

dir_list = ['demo_sequences']

root_data_dir = 'data/sequences'
old_ckpt_path = 'checkpoint/ood_ib1_0.2_decay_10_wd_0_22_last.tar'
result_path = 'results/ood_prob_dict.pickle'
error_list_path = 'results/ood_error_list.pickle'
model_arch = 'resnet50'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 299
batch_size = 64


class MyCrop:
    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, img):
        return TF.crop(img, self.top, self.left, self.height, self.width)


class SingleImageDatasetModified(Dataset):
    def __init__(self, dir_list, transform, latest_prob_dict):
        self.path_list = []
        self.transform = transform

        for subdir in dir_list:
            data_dir = join(root_data_dir, subdir)
            for folder in os.listdir(data_dir):
                idx = folder.split('_')[0]
                folder_dir = join(data_dir, folder)
                for f in os.listdir(folder_dir):
                    if not f[-4:] == '.png':
                        continue
                    if idx in latest_prob_dict and f in latest_prob_dict[idx]:
                        continue
                    self.path_list.append((subdir, folder, f))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        subdir, folder, fname = self.path_list[index]
        image_path = join(root_data_dir, subdir, folder, fname)
        idx = folder.split('_')[0]
        img = Image.open(image_path)
        if not img.mode == 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, idx, fname


transform_test = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        MyCrop(17, 0, 240, 299),
        # transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


if __name__ == '__main__':
    # load existing prob dict or initialize a new one
    if exists(result_path):
        with open(result_path, 'rb') as f:
            prob_dict = pickle.load(f)
    else:
        prob_dict = {}

    # load existing error list or initialize a new one
    if exists(error_list_path):
        with open(error_list_path, 'rb') as f:
            error_list = pickle.load(f)
    else:
        error_list = []

    # dataloader
    dataset_pred = SingleImageDatasetModified(dir_list, transform=transform_test, latest_prob_dict=prob_dict)
    print('Dataset size: ' + str(len(dataset_pred)))
    dataloader_pred = DataLoader(dataset_pred, batch_size=batch_size, shuffle=False, num_workers=4)

    # model
    if model_arch == 'resnet18':
        model = resnet18(num_classes=2)
    elif model_arch == 'resnet34':
        model = resnet34(num_classes=2)
    elif model_arch == 'resnet50':
        model = resnet50(num_classes=2)
    elif model_arch == 'inception':
        model = Inception3(num_classes=2, aux_logits=True, transform_input=False)
    else:
        raise
    model = model.to(device)

    # load old parameters
    checkpoint = torch.load(old_ckpt_path, map_location=device)
    if old_ckpt_path[-4:] == '.tar':  # it is a checkpoint dictionary rather than just model parameters
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print('Old checkpoint loaded: ' + old_ckpt_path)

    model.eval()
    # run
    count = 0
    for inputs, idx_list, fname_list in tqdm(dataloader_pred):
        try:
            inputs = inputs.to(device)
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                prob = torch.sigmoid(outputs)
            prob_list = prob.cpu().numpy()
            for i in range(len(idx_list)):
                idx = idx_list[i]
                fname = fname_list[i]
                prob_sample = prob_list[i]

                if not idx in prob_dict:
                    prob_dict[idx] = {}
                prob_dict[idx][fname] = prob_sample

        except:  # take a note on the batch that causes error
            error_list.append((idx_list, fname_list))
        if count % 200 == 0:
            with open(join(result_path), 'wb') as f:
                pickle.dump(prob_dict, f)
            with open(join(error_list_path), 'wb') as f:
                pickle.dump(error_list, f)
        count += 1

    with open(join(result_path), 'wb') as f:
        pickle.dump(prob_dict, f)
    with open(join(error_list_path), 'wb') as f:
        pickle.dump(error_list, f)

    print('Done!')


