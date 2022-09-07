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
from torchvision.models import Inception3

from utils.image_dataset import *
from LR_models.siamese_model_rgb import *

"""
This script is for generating the prediction scores of LR model for images in sequences.
A sequence of images are stored in a folder. An image is named by the year of the image 
plus an auxiliary index. E.g., '2007_0.png', '2007_1.png', '2008_0.png'.
"""

dir_list = ['demo_sequences']

root_data_dir = 'data/sequences'
old_ckpt_path = 'checkpoint/LR_nconvs_3_depth_128_nfilters_512_33_last.tar'
result_path = 'results/LR_prob_dict.pickle'
error_list_path = 'results/LR_error_list.pickle'
anchor_images_dict_path = 'results/anchor_images_dict.pickle'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
backbone = 'resnet34'
input_size = 299
batch_size = 64

transform_test = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        # transforms.Lambda(mask_image_info),
        transforms.ToTensor()
    ])


class ImagePairDatasetModified(Dataset):
    def __init__(self, dir_list, anchor_images_dict, transform, latest_prob_dict):
        self.couple_list = []
        self.transform = transform

        for subdir in dir_list:
            data_dir = join(root_data_dir, subdir)
            for folder in os.listdir(data_dir):
                idx = folder.split('_')[0]
                folder_dir = join(data_dir, folder)
                if idx not in anchor_images_dict:
                    continue
                anchor_images = anchor_images_dict[idx]
                for anchor_f in anchor_images:
                    for tar_f in os.listdir(folder_dir):
                        if not tar_f[-4:] == '.png':
                            continue
                        if idx in latest_prob_dict and anchor_f in latest_prob_dict[idx] and tar_f in latest_prob_dict[idx][anchor_f]:
                            continue
                        self.couple_list.append((subdir, folder, anchor_f, tar_f))

    def __len__(self):
        return len(self.couple_list)

    def __getitem__(self, index):
        subdir, folder, anchor_f, tar_f = self.couple_list[index]
        ref_img_path = join(root_data_dir, subdir, folder, anchor_f)
        tar_img_path = join(root_data_dir, subdir, folder, tar_f)
        idx = folder.split('_')[0]

        img_ref = Image.open(ref_img_path)
        img_tar = Image.open(tar_img_path)
        if not img_ref.mode == 'RGB':
            img_ref = img_ref.convert('RGB')
        if not img_tar.mode == 'RGB':
            img_tar = img_tar.convert('RGB')

        img_ref = self.transform(img_ref)
        img_tar = self.transform(img_tar)

        return img_ref, img_tar, idx, anchor_f, tar_f


if __name__ == '__main__':
    # load anchor_images_dict
    with open(anchor_images_dict_path, 'rb') as f:
        anchor_images_dict = pickle.load(f)

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
    dataset_pred = ImagePairDatasetModified(dir_list, anchor_images_dict, transform=transform_test, latest_prob_dict=prob_dict)
    print('Dataset size: ' + str(len(dataset_pred)))
    dataloader_pred = DataLoader(dataset_pred, batch_size=batch_size, shuffle=False, num_workers=4)

    # model
    model = psn_depthwise_cc_layerwise_3layers_l234(backbone=backbone, nconvs=3, depth=128, nfilters=512, kernel_size=3)
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
    for inputs_ref, inputs_tar, idx_list, anchor_f_list, tar_f_list in tqdm(dataloader_pred):
        try:
            inputs_ref = inputs_ref.to(device)
            inputs_tar = inputs_tar.to(device)
            with torch.set_grad_enabled(False):
                outputs = model(inputs_tar, inputs_ref)
                prob = F.softmax(outputs, dim=1)
            pos_prob_list = prob[:, 1].cpu().numpy()
            for i in range(len(idx_list)):
                idx = idx_list[i]
                anchor_f = anchor_f_list[i]
                tar_f = tar_f_list[i]
                pos_prob = pos_prob_list[i]

                if not idx in prob_dict:
                    prob_dict[idx] = {}
                if not anchor_f in prob_dict[idx]:
                    prob_dict[idx][anchor_f] = {}

                prob_dict[idx][anchor_f][tar_f] = pos_prob

        except:  # take a note on the batch that causes error
            error_list.append((idx_list, anchor_f_list, tar_f_list))

        if count % 400 == 0:
            with open(result_path, 'wb') as f:
                pickle.dump(prob_dict, f)
            with open(error_list_path, 'wb') as f:
                pickle.dump(error_list, f)
        count += 1

    with open(result_path, 'wb') as f:
        pickle.dump(prob_dict, f)
    with open(error_list_path, 'wb') as f:
        pickle.dump(error_list, f)

    print('Done!')
