# -*-coding:utf-8 -*-

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


import torch.nn.functional as F
from torchvision.models import Inception3, resnet18, ResNet
from collections import namedtuple

_InceptionOuputs = namedtuple('InceptionOuputs', ['logits', 'aux_logits'])


class Inception3_modified(Inception3):
    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        fm35 = self.Mixed_5d(x)

        # N x 288 x 35 x 35
        # fm35 = x.clone()
        x = self.Mixed_6a(fm35)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        fm8 = self.Mixed_7c(x)

        # return x, fm8
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x2048 = F.adaptive_avg_pool2d(fm8, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x2048, training=self.training)
        # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 2048
        out = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:
            return _InceptionOuputs(x, aux)
        return out, fm8, x2048


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class resnet18_modified(ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # 64 x 75 x 75
        fm38 = self.layer2(x)
        # 128 x 38 x 38
        fm19 = self.layer3(fm38)
        # 256 x 19 x 19
        # fm10 = self.layer4(fm19)
        # 512 x 10 x 10
        x256 = self.avgpool(fm19)
        x = x256.view(x256.size(0), -1)

        # x512 = self.avgpool(fm10)
        # x = x512.view(x512.size(0), -1)
        # x = self.fc(x)

        return x, fm19
        # return x256, fm19

class naive_model(nn.Module):
    def __init__(self):
        super(naive_model, self).__init__()
        self.net1 = resnet18_modified(BasicBlock, [2, 2, 2, 2])
        fc_features = self.net1.fc.in_features
        self.net1.fc = nn.Linear(256, 128)

        self.fc1 = nn.Linear(361, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, img1, img2):
        img1 = torch.cat([img1, img1, img1], 1)
        img2 = torch.cat([img2, img2, img2], 1)

        x1, fm1 = self.net1(img1)
        x2, fm2 = self.net1(img2)

        # mix_fm = F.conv2d(fm1[0, :, :, :].unsqueeze(0), fm2[0, :, :, :].unsqueeze(0), padding=9, stride=1)
        # new_vec = mix_fm.view(1, -1)
        #
        # for i in range(1, fm1.shape[0]):
        #     new_fm = F.conv2d(fm1[i, :, :, :].unsqueeze(0), fm2[i, :, :, :].unsqueeze(0), padding=9, stride=1)
        #     new_fm = new_fm.view(1, -1)
        #     new_vec = torch.cat((new_vec, new_fm), 0)

        fm1 = fm1.reshape(-1, fm1.size()[2], fm1.size()[3])
        fm1 = fm1.unsqueeze(0)
        new_vec = F.conv2d(fm1, fm2, padding=9, stride=1, groups=fm2.size()[0]).permute(1, 0, 2, 3)

        new_vec = new_vec.view(new_vec.size(0), -1)
        out = F.relu(new_vec)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
