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

old_ckpt_path = 'checkpoint/LR_rgb/LR_012_res34_dwcc_tr_nomask_l234_augmented_2/LR_nconvs_3_depth_128_nfilters_512_33_last.tar'
result_dir = 'results/massive_test_set_reproduced'
anchor_images_dict_path = join(result_dir, 'anchor_images_dict.pickle')

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
    def __init__(self, anchor_images_dict, sequence_data_dirs, transform):
        self.couple_list = []
        self.transform = transform

        for data_dir in sequence_data_dirs:
            for folder in os.listdir(data_dir):
                county, idx, install_year = folder.split('_')
                folder_dir = join(data_dir, folder)
                anchor_images = anchor_images_dict[idx]
                for anchor_f in anchor_images:
                    ref_path = join(folder_dir, anchor_f)
                    for tar_f in os.listdir(folder_dir):
                        if not tar_f[-4:] == '.png':
                            continue
                        tar_path = join(folder_dir, tar_f)
                        self.couple_list.append((ref_path, tar_path, idx, anchor_f, tar_f))

    def __len__(self):
        return len(self.couple_list)

    def __getitem__(self, index):
        each_couple_list = self.couple_list[index]
        img_ref = Image.open(each_couple_list[0])
        img_tar = Image.open(each_couple_list[1])
        if not img_ref.mode == 'RGB':
            img_ref = img_ref.convert('RGB')
        if not img_tar.mode == 'RGB':
            img_tar = img_tar.convert('RGB')

        img_ref = self.transform(img_ref)
        img_tar = self.transform(img_tar)

        idx = each_couple_list[2]
        anchor_f = each_couple_list[3]
        tar_f = each_couple_list[4]
        return img_ref, img_tar, idx, anchor_f, tar_f


if __name__ == '__main__':
    # anchor_images_dict
    with open(anchor_images_dict_path, 'rb') as f:
        anchor_images_dict = pickle.load(f)

    # dataloader
    dataset_pred = ImagePairDatasetModified(anchor_images_dict, data_dirs, transform=transform_test)
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
    prob_dict = dict()
    for inputs_ref, inputs_tar, idx_list, anchor_f_list, tar_f_list in tqdm(dataloader_pred):
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

    with open(join(result_dir, 'LR_prob_dict_l234_augmented.pickle'), 'wb') as f:
        pickle.dump(prob_dict, f)

