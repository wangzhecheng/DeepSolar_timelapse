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


class sn_depthwise_cc(nn.Module):
    def __init__(self, nconvs=2, nfilters=256):
        super(sn_depthwise_cc, self).__init__()
        self.net1 = resnet18_modified(BasicBlock, [2, 2, 2, 2])
        self.net1.fc = nn.Linear(256, 128)

        self.nconvs = nconvs
        assert self.nconvs in [0, 1, 2, 3]

        if self.nconvs in [1, 2, 3]:
            self.conv_combo_1 = nn.Sequential(nn.Conv2d(
                256, nfilters, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs in [2, 3]:
            self.conv_combo_2 = nn.Sequential(nn.Conv2d(
                nfilters, nfilters, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(nfilters),
                nn.ReLU(inplace=True))

        if self.nconvs == 3:
            self.conv_combo_3 = nn.Sequential(nn.Conv2d(
                nfilters, nfilters, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(nfilters),
                nn.ReLU(inplace=True))

        if self.nconvs == 0:
            self.fc = nn.Linear(256, 2)
        else:
            self.fc = nn.Linear(nfilters, 2)

    def forward(self, img1, img2):
        img1 = torch.cat([img1, img1, img1], 1)
        img2 = torch.cat([img2, img2, img2], 1)

        x1, fm1 = self.net1(img1)
        x2, fm2 = self.net1(img2)

        # depth-wise cross correlation
        nchannels = fm1.size()[1]
        fm1 = fm1.reshape(-1, fm1.size()[2], fm1.size()[3])
        fm1 = fm1.unsqueeze(0)
        fm2 = fm2.reshape(-1, fm2.size()[2], fm2.size()[3])
        fm2 = fm2.unsqueeze(0).permute(1, 0, 2, 3)
        new_vec = F.conv2d(fm1, fm2, padding=9, stride=1, groups=fm2.size()[0]).squeeze()
        new_vec = new_vec.reshape(-1, nchannels, new_vec.size()[1], new_vec.size()[2])
        out = F.relu(new_vec)

        # convolution layers
        if self.nconvs in [1, 2, 3]:
            out = self.conv_combo_1(out)
        if self.nconvs in [2, 3]:
            out = self.conv_combo_2(out)
        if self.nconvs == 3:
            out = self.conv_combo_3(out)

        # global average pooling
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class sn_depthwise_cc_1x1(nn.Module):
    def __init__(self, nconvs=2, nfilters=256):
        super(sn_depthwise_cc_1x1, self).__init__()
        self.net1 = resnet18_modified(BasicBlock, [2, 2, 2, 2])
        self.net1.fc = nn.Linear(256, 128)

        self.nconvs = nconvs
        assert self.nconvs in [0, 1, 2, 3]

        if self.nconvs in [1, 2, 3]:
            self.conv_combo_1 = nn.Sequential(nn.Conv2d(
                256, nfilters, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs in [2, 3]:
            self.conv_combo_2 = nn.Sequential(nn.Conv2d(
                nfilters, nfilters, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs == 3:
            self.conv_combo_3 = nn.Sequential(nn.Conv2d(
                nfilters, nfilters, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs == 0:
            self.fc = nn.Linear(256, 2)
        else:
            self.fc = nn.Linear(nfilters, 2)

    def forward(self, img1, img2):
        img1 = torch.cat([img1, img1, img1], 1)
        img2 = torch.cat([img2, img2, img2], 1)

        x1, fm1 = self.net1(img1)
        x2, fm2 = self.net1(img2)

        # depth-wise cross correlation
        nchannels = fm1.size()[1]
        fm1 = fm1.reshape(-1, fm1.size()[2], fm1.size()[3])
        fm1 = fm1.unsqueeze(0)
        fm2 = fm2.reshape(-1, fm2.size()[2], fm2.size()[3])
        fm2 = fm2.unsqueeze(0).permute(1, 0, 2, 3)
        new_vec = F.conv2d(fm1, fm2, padding=9, stride=1, groups=fm2.size()[0]).squeeze()
        new_vec = new_vec.reshape(-1, nchannels, new_vec.size()[1], new_vec.size()[2])
        out = F.relu(new_vec)

        # convolution layers
        if self.nconvs in [1, 2, 3]:
            out = self.conv_combo_1(out)
        if self.nconvs in [2, 3]:
            out = self.conv_combo_2(out)
        if self.nconvs == 3:
            out = self.conv_combo_3(out)

        # global average pooling
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

