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

data_dirs = ['/home/ubuntu/projects/data/deepsolar2/cleaned/sequence_1/val',
             '/home/ubuntu/projects/data/deepsolar2/cleaned/sequence_2/val',
             '/home/ubuntu/projects/data/deepsolar2/cleaned/sequence_0/test',
             '/home/ubuntu/projects/data/deepsolar2/cleaned/sequence_1/test',
             '/home/ubuntu/projects/data/deepsolar2/cleaned/sequence_2/test',
             '/home/ubuntu/projects/data/deepsolar2/cleaned/sequence_3/test']

old_ckpt_path = 'checkpoint/HR/HR_1_HR_2_ft_all_hp_search/HR_decay_10_lr_0.0001_8_last.tar'
result_dir = 'results/massive_test_set_reproduced'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 299

transform_test = transforms.Compose([
                 transforms.Resize(input_size),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                 ])

if __name__ == '__main__':
    # model
    model = Inception3(num_classes=2, aux_logits=True, transform_input=False)
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
            ds = SequenceDataset(folder_dir, transform=transform_test)
            dl = DataLoader(ds, shuffle=False, batch_size=len(ds))
            img, f_list = next(iter(dl))
            img = img.to(device)
            with torch.set_grad_enabled(False):
                outputs = model(img)
                prob = F.softmax(outputs, dim=1)[:, 1]
            prob = prob.cpu().numpy()
            for i, f in enumerate(f_list):
                prob_dict[idx][f] = prob[i]

    with open(join(result_dir, 'HR_prob_dict_HR_12_distorted_metrics.pickle'), 'wb') as f:
        pickle.dump(prob_dict, f)
