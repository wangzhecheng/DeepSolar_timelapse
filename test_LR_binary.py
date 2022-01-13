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
from torchvision.models import Inception3

from utils.image_dataset import *
# from LR_models.couple_model import *
# from LR_models.siamese_model_binary import *
from LR_models.siamese_model_binary_layerwise import *

# Configuration
# directory for loading training/validation/test data
# data_dir = '/home/ubuntu/projects/deepsolar/dataset/HR/test'
data_dirs = [
             '/home/ubuntu/projects/data/deepsolar2/cleaned/LR_0_binary/test',
             '/home/ubuntu/projects/data/deepsolar2/cleaned/LR_1_binary/test',
             '/home/ubuntu/projects/data/deepsolar2/cleaned/LR_2_binary/test'
             ]
reference_mapping_paths = [
                           '/home/ubuntu/projects/data_preparation/reference_mapping/reference_mapping_LR_0.pickle',
                           '/home/ubuntu/projects/data_preparation/reference_mapping/reference_mapping_LR_1_single.pickle',
                           '/home/ubuntu/projects/data_preparation/reference_mapping/reference_mapping_LR_2_single.pickle'
                           ]

old_ckpt_path = 'checkpoint/LR/LR_012_dwcc_3x3_layerwise/LR_nconvs_3_lr_0.03_nfilters_512_29_last.tar'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 299
batch_size = 32
# threshold = 0.5  # threshold probability to identify am image as positive
threshold_list = np.linspace(0.0, 1.0, 101).tolist() + [0.31]

def metrics(stats):
    """stats: {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
    return: must be a single number """
    # precision = (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FP'] + 0.00001)
    # recall = (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FN'] + 0.00001)
    # F1 = 2.0 * stats['TP'] / (2 * stats['TP'] + stats['FP'] + stats['FN'])
    spec = (stats['TN'] + 0.00001) * 1.0 / (stats['TN'] + stats['FP'] + 0.00001)
    sens = (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FN'] + 0.00001)
    return 2.0 * spec * sens / (spec + sens + 1e-7)


def test_model(model, dataloader, metrics, threshold_list):
    stats = {x: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for x in threshold_list}
    metric_values = {}
    model.eval()
    for inputs_ref, inputs_tar, labels in tqdm(dataloader):
        inputs_ref = inputs_ref.to(device)
        inputs_tar = inputs_tar.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs_ref, inputs_tar)
            prob = F.softmax(outputs, dim=1)
            for threshold in threshold_list:
                preds = prob[:, 1] >= threshold
                stats[threshold]['TP'] += torch.sum((preds == 1) * (labels == 1)).cpu().item()
                stats[threshold]['TN'] += torch.sum((preds == 0) * (labels == 0)).cpu().item()
                stats[threshold]['FP'] += torch.sum((preds == 1) * (labels == 0)).cpu().item()
                stats[threshold]['FN'] += torch.sum((preds == 0) * (labels == 1)).cpu().item()

    for threshold in threshold_list:
        metric_values[threshold] = metrics(stats[threshold])

    return stats, metric_values

transform_test = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])

if __name__ == '__main__':
    # data
    dataset_test = ImagePairDataset(data_dirs, reference_mapping_paths, is_train=False, binary=True,
                                    transform=transform_test)
    print('Dataset size: ' + str(len(dataset_test)))
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
    # model
    # model = naive_model()
    model = sn_depthwise_cc_layerwise_3x3(nconvs=3, nfilters=512)
    model = model.to(device)
    # load old parameters
    checkpoint = torch.load(old_ckpt_path, map_location=device)
    if old_ckpt_path[-4:] == '.tar':  # it is a checkpoint dictionary rather than just model parameters
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print('Old checkpoint loaded: ' + old_ckpt_path)


    best_threshold = None
    best_metric_value = 0.

    stats, metric_values = test_model(model, dataloader_test, metrics, threshold_list=threshold_list)

    for threshold in threshold_list:
        spec = (stats[threshold]['TN'] + 0.00001) * 1.0 / (stats[threshold]['TN'] + stats[threshold]['FP'] + 0.00001)
        sens = (stats[threshold]['TP'] + 0.00001) * 1.0 / (stats[threshold]['TP'] + stats[threshold]['FN'] + 0.00001)
        metric_value = metric_values[threshold]
        print('Threshold: ' + str(threshold) + ' metric value: ' + str(metric_value) + ' sensitivity: ' + str(
            round(sens, 4)) + ' specificity: ' + str(round(spec, 4)))
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = threshold

    print('Best threshold: ' + str(best_threshold) + ' best metric value: ' + str(best_metric_value))
