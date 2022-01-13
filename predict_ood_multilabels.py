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
import skimage
import skimage.io
import skimage.transform
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
from utils.mk_cam import *
from utils.mk_mask import *
from utils.naive_method import *
from LR_models.siamese_model_rgb import *

data_dirs = ['/home/ubuntu/projects/data/deepsolar2/cleaned/sequence_1/val',
             '/home/ubuntu/projects/data/deepsolar2/cleaned/sequence_2/val',
             '/home/ubuntu/projects/data/deepsolar2/cleaned/sequence_0/test',
             '/home/ubuntu/projects/data/deepsolar2/cleaned/sequence_1/test',
             '/home/ubuntu/projects/data/deepsolar2/cleaned/sequence_2/test',
             '/home/ubuntu/projects/data/deepsolar2/cleaned/sequence_3/test']

old_ckpt_path = 'checkpoint/ood/ood_all_LR_012_HR_12_resnet50_multilabels/ood_ib1_0.2_decay_10_wd_0_22_last.tar'
# old_ckpt_path = 'checkpoint/ood/ood_all_LR_012_HR_12_resnet50_multilabels_change_overall_metrics/ood_lr_0.0001_decay_4_wd_0_9_last.tar'
result_dir = 'results/massive_test_set_reproduced'
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
    def __init__(self, sequence_data_dirs, transform):
        self.path_list = []
        self.transform = transform

        for data_dir in sequence_data_dirs:
            for folder in os.listdir(data_dir):
                county, idx, install_year = folder.split('_')
                folder_dir = join(data_dir, folder)
                for f in os.listdir(folder_dir):
                    if not f[-4:] == '.png':
                        continue
                    image_path = join(folder_dir, f)
                    self.path_list.append((image_path, idx, f))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        image_path, idx, fname = self.path_list[index]
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

    # dataloader
    dataset_pred = SingleImageDatasetModified(data_dirs, transform=transform_test)
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
    prob_dict = dict()
    for inputs, idx_list, fname_list in tqdm(dataloader_pred):
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

    with open(join(result_dir, 'ood_res50_multilabels.pickle'), 'wb') as f:
        pickle.dump(prob_dict, f)

