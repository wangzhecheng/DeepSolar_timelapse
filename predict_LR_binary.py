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
from LR_models.couple_model import *

data_dirs = ['/home/ubuntu/projects/data/deepsolar2/cleaned/sequence_1/val',
             '/home/ubuntu/projects/data/deepsolar2/cleaned/sequence_2/val',
             '/home/ubuntu/projects/data/deepsolar2/cleaned/sequence_0/test',
             '/home/ubuntu/projects/data/deepsolar2/cleaned/sequence_1/test',
             '/home/ubuntu/projects/data/deepsolar2/cleaned/sequence_2/test']

old_ckpt_path = 'checkpoint/LR/baseline/fm_conv_pad9_2fc_61_last.tar'
result_dir = 'results/baseline'
anchor_images_dict_path = join(result_dir, 'anchor_images_dict.pickle')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 299

transform_test = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])


def test_model(model, img_ref, img_tar):
    inputs_ref = img_ref
    inputs_tar = img_tar
    inputs_ref = inputs_ref.to(device)
    inputs_tar = inputs_tar.to(device)
    with torch.set_grad_enabled(False):
        outputs = model(inputs_ref, inputs_tar)
        # outputs, _ = model(inputs_tar)
        prob = F.softmax(outputs, dim=1)
    return prob[:, 1]


if __name__ == '__main__':
    # anchor_images_dict
    with open(anchor_images_dict_path, 'rb') as f:
        anchor_images_dict = pickle.load(f)
    # binarizor
    bi_trans = NaiveMethod()
    # model
    model = naive_model()
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
    for data_dir in data_dirs:
        print(data_dir)
        for folder in tqdm(os.listdir(data_dir)):
            county, idx, install_year = folder.split('_')
            prob_dict[idx] = {}
            folder_dir = join(data_dir, folder)
            anchor_images = anchor_images_dict[idx]
            for anchor_f in anchor_images:
                prob_dict[idx][anchor_f] = {}
                img_ref, large_mask = bi_trans.perform_extract(join(folder_dir, anchor_f), is_ref=1)
                img_ref = Image.fromarray(img_ref)
                img_ref = transform_test(img_ref)

                for tar_f in os.listdir(folder_dir):
                    if not tar_f[-4:] == '.png':
                        continue
                    img_tar = bi_trans.perform_extract(join(folder_dir, tar_f), is_ref=0, large_mask=large_mask)
                    img_tar = Image.fromarray(img_tar)
                    img_tar = transform_test(img_tar)
                    pos_prob = test_model(model, img_ref.unsqueeze(0), img_tar.unsqueeze(0)).cpu().numpy()[0]

                    prob_dict[idx][anchor_f][tar_f] = pos_prob

    with open(join(result_dir, 'LR_prob_dict.pickle'), 'wb') as f:
        pickle.dump(prob_dict, f)

