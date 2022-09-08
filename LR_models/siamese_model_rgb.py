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


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class resnet_modified(ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        fm75 = self.layer1(x)
        # 64 x 75 x 75
        fm38 = self.layer2(fm75)
        # 128 x 38 x 38
        fm19 = self.layer3(fm38)
        # 256 x 19 x 19
        fm10 = self.layer4(fm19)
        # 512 x 10 x 10
        # x256 = self.avgpool(fm19)
        # x = x256.view(x256.size(0), -1)

        # x512 = self.avgpool(fm10)
        # x = x512.view(x512.size(0), -1)
        # x = self.fc(x)

        return fm75, fm38, fm19, fm10


def XCorr_depthwise(fm1, fm2, padding):
    """
    :param fm1: inputs
    :param fm2: filters
    :param padding: padding in cross correlation layer
    :return: similarity map after cross correlation
    """
    nchannels = fm1.size()[1]
    fm1 = fm1.reshape(-1, fm1.size()[2], fm1.size()[3])
    fm1 = fm1.unsqueeze(0)
    fm2 = fm2.reshape(-1, fm2.size()[2], fm2.size()[3])
    fm2 = fm2.unsqueeze(0).permute(1, 0, 2, 3)
    out = F.conv2d(fm1, fm2, padding=padding, stride=1, groups=fm2.size()[0]).squeeze()
    out = out.reshape(-1, nchannels, out.size()[1], out.size()[2])
    return out


class sn_cc_l3(nn.Module):
    def __init__(self, hidden=128, backbone='resnet18'):
        super(sn_cc_l3, self).__init__()
        assert backbone in ['resnet18', 'resnet34', 'resnet50']
        if backbone == 'resnet18':
            self.net1 = resnet_modified(BasicBlock, [2, 2, 2, 2])
        elif backbone == 'resnet34':
            self.net1 = resnet_modified(BasicBlock, [3, 4, 6, 3])
        elif backbone == 'resnet50':
            self.net1 = resnet_modified(Bottleneck, [3, 4, 6, 3])
        else:
            raise
        self.backbone = backbone
        self.nchannels_dict = {'resnet18': {1: 64, 2: 128, 3: 256, 4: 512},
                               'resnet34': {1: 64, 2: 128, 3: 256, 4: 512},
                               'resnet50': {1: 256, 2: 512, 3: 1024, 4: 2048}}

        self.fc1 = nn.Linear(361, hidden)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, img1, img2):

        _, _, fm1, _ = self.net1(img1)
        _, _, fm2, _ = self.net1(img2)

        fm1 = fm1.reshape(-1, fm1.size()[2], fm1.size()[3])
        fm1 = fm1.unsqueeze(0)
        new_vec = F.conv2d(fm1, fm2, padding=9, stride=1, groups=fm2.size()[0]).permute(1, 0, 2, 3)

        new_vec = new_vec.view(new_vec.size(0), -1)
        out = F.relu(new_vec)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


class psn_cc_l3(nn.Module):
    def __init__(self, hidden=128, backbone='resnet18'):
        super(psn_cc_l3, self).__init__()
        assert backbone in ['resnet18', 'resnet34', 'resnet50']
        if backbone == 'resnet18':
            self.net1 = resnet_modified(BasicBlock, [2, 2, 2, 2])
            self.net2 = resnet_modified(BasicBlock, [2, 2, 2, 2])
        elif backbone == 'resnet34':
            self.net1 = resnet_modified(BasicBlock, [3, 4, 6, 3])
            self.net2 = resnet_modified(BasicBlock, [3, 4, 6, 3])
        elif backbone == 'resnet50':
            self.net1 = resnet_modified(Bottleneck, [3, 4, 6, 3])
            self.net2 = resnet_modified(Bottleneck, [3, 4, 6, 3])
        else:
            raise
        self.backbone = backbone
        self.nchannels_dict = {'resnet18': {1: 64, 2: 128, 3: 256, 4: 512},
                               'resnet34': {1: 64, 2: 128, 3: 256, 4: 512},
                               'resnet50': {1: 256, 2: 512, 3: 1024, 4: 2048}}

        self.fc1 = nn.Linear(361, hidden)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, img1, img2):

        _, _, fm1, _ = self.net1(img1)
        _, _, fm2, _ = self.net2(img2)

        fm1 = fm1.reshape(-1, fm1.size()[2], fm1.size()[3])
        fm1 = fm1.unsqueeze(0)
        new_vec = F.conv2d(fm1, fm2, padding=9, stride=1, groups=fm2.size()[0]).permute(1, 0, 2, 3)

        new_vec = new_vec.view(new_vec.size(0), -1)
        out = F.relu(new_vec)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


class psn_depthwise_cc_l3(nn.Module):
    def __init__(self, backbone='resnet18', nconvs=2, nfilters=256, kernel_size=3):
        super(psn_depthwise_cc_l3, self).__init__()
        assert backbone in ['resnet18', 'resnet34', 'resnet50']
        if backbone == 'resnet18':
            self.net1 = resnet_modified(BasicBlock, [2, 2, 2, 2])
            self.net2 = resnet_modified(BasicBlock, [2, 2, 2, 2])
        elif backbone == 'resnet34':
            self.net1 = resnet_modified(BasicBlock, [3, 4, 6, 3])
            self.net2 = resnet_modified(BasicBlock, [3, 4, 6, 3])
        elif backbone == 'resnet50':
            self.net1 = resnet_modified(Bottleneck, [3, 4, 6, 3])
            self.net2 = resnet_modified(Bottleneck, [3, 4, 6, 3])
        else:
            raise
        self.backbone = backbone
        self.nchannels_dict = {'resnet18': {1: 64, 2: 128, 3: 256, 4: 512},
                               'resnet34': {1: 64, 2: 128, 3: 256, 4: 512},
                               'resnet50': {1: 256, 2: 512, 3: 1024, 4: 2048}}

        self.nconvs = nconvs
        assert self.nconvs in [0, 1, 2, 3]

        self.nchannels = self.nchannels_dict[self.backbone][3]

        if kernel_size == 3:
            padding = 1
        elif kernel_size == 1:
            padding = 0
        else:
            raise

        if self.nconvs in [1, 2, 3]:
            self.conv_combo_1 = nn.Sequential(nn.Conv2d(
                self.nchannels, nfilters, kernel_size=kernel_size, padding=padding, stride=1),
                nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs in [2, 3]:
            self.conv_combo_2 = nn.Sequential(nn.Conv2d(
                nfilters, nfilters, kernel_size=kernel_size, padding=padding, stride=1), nn.BatchNorm2d(nfilters),
                nn.ReLU(inplace=True))

        if self.nconvs == 3:
            self.conv_combo_3 = nn.Sequential(nn.Conv2d(
                nfilters, nfilters, kernel_size=kernel_size, padding=padding, stride=1), nn.BatchNorm2d(nfilters),
                nn.ReLU(inplace=True))

        if self.nconvs == 0:
            self.fc = nn.Linear(self.nchannels, 2)
        else:
            self.fc = nn.Linear(nfilters, 2)

    def forward(self, img1, img2):

        _, _, fm1, _ = self.net1(img1)
        _, _, fm2, _ = self.net2(img2)

        # depth-wise cross correlation
        nchannels = fm1.size()[1]
        fm1 = fm1.reshape(-1, fm1.size()[2], fm1.size()[3])
        fm1 = fm1.unsqueeze(0)
        fm2 = fm2.reshape(-1, fm2.size()[2], fm2.size()[3])
        fm2 = fm2.unsqueeze(0).permute(1, 0, 2, 3)
        new_vec = F.conv2d(fm1, fm2, padding=9, stride=1, groups=fm2.size()[0]).squeeze()
        new_vec = new_vec.reshape(-1, nchannels, new_vec.size()[1], new_vec.size()[2])
        out = F.relu(new_vec)  # batch_size * nchannels * H * W

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


class psn_depthwise_cc_layerwise_l23(nn.Module):
    def __init__(self, backbone='resnet18', nconvs=2, depth=128, nfilters=256, kernel_size=3):
        super(psn_depthwise_cc_layerwise_l23, self).__init__()
        assert backbone in ['resnet18', 'resnet34', 'resnet50']
        if backbone == 'resnet18':
            self.net1 = resnet_modified(BasicBlock, [2, 2, 2, 2])
            self.net2 = resnet_modified(BasicBlock, [2, 2, 2, 2])
        elif backbone == 'resnet34':
            self.net1 = resnet_modified(BasicBlock, [3, 4, 6, 3])
            self.net2 = resnet_modified(BasicBlock, [3, 4, 6, 3])
        elif backbone == 'resnet50':
            self.net1 = resnet_modified(Bottleneck, [3, 4, 6, 3])
            self.net2 = resnet_modified(Bottleneck, [3, 4, 6, 3])
        else:
            raise
        self.backbone = backbone
        self.nchannels_dict = {'resnet18': {1: 64, 2: 128, 3: 256, 4: 512},
                               'resnet34': {1: 64, 2: 128, 3: 256, 4: 512},
                               'resnet50': {1: 256, 2: 512, 3: 1024, 4: 2048}}

        self.extra_1x1_conv_1 = nn.Conv2d(self.nchannels_dict[self.backbone][2], depth, kernel_size=1, stride=1,
                                          padding=0, bias=False)
        self.extra_1x1_conv_2 = nn.Conv2d(self.nchannels_dict[self.backbone][3], depth, kernel_size=1, stride=1,
                                          padding=0, bias=False)

        self.depth = depth
        self.nconvs = nconvs
        assert self.nconvs in [0, 1, 2, 3]

        if kernel_size == 3:
            padding = 1
        elif kernel_size == 1:
            padding = 0
        else:
            raise

        if self.nconvs in [1, 2, 3]:
            self.conv_combo_1 = nn.Sequential(nn.Conv2d(
                depth*2, nfilters, kernel_size=kernel_size, padding=padding, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs in [2, 3]:
            self.conv_combo_2 = nn.Sequential(nn.Conv2d(
                nfilters, nfilters, kernel_size=kernel_size, padding=padding, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs == 3:
            self.conv_combo_3 = nn.Sequential(nn.Conv2d(
                nfilters, nfilters, kernel_size=kernel_size, padding=padding, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs == 0:
            self.fc = nn.Linear(depth*2, 2)
        else:
            self.fc = nn.Linear(nfilters, 2)

    def forward(self, img1, img2):

        _, fm38_1, fm19_1, _ = self.net1(img1)
        _, fm38_2, fm19_2, _ = self.net2(img2)

        # change the number of channels
        fm38_1 = self.extra_1x1_conv_1(fm38_1)  # batch_size * depth * 38 * 38
        fm38_2 = self.extra_1x1_conv_1(fm38_2)  # batch_size * depth * 38 * 38

        fm19_1 = self.extra_1x1_conv_2(fm19_1)  # batch_size * depth * 19 * 19
        fm19_2 = self.extra_1x1_conv_2(fm19_2)  # batch_size * depth * 19 * 19

        # depth-wise cross correlation
        sm38 = XCorr_depthwise(fm38_1, fm38_2, padding=19)  # batch_size * depth * 38 * 38
        sm19 = XCorr_depthwise(fm19_1, fm19_2, padding=9)   # batch_size * depth * 19 * 19

        # aggregate along the channel
        sm38_downsampled = F.adaptive_avg_pool2d(sm38, (sm19.size()[2], sm19.size()[3]))  # batch_size * depth * 19 * 19
        out = torch.cat([sm38_downsampled, sm19], 1)  # batch_size * (2*depth) * 19 * 19
        out = F.relu(out)

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


class psn_depthwise_cc_layerwise_l34(nn.Module):
    def __init__(self, backbone='resnet18', nconvs=2, depth=128, nfilters=256, kernel_size=3):
        super(psn_depthwise_cc_layerwise_l34, self).__init__()
        assert backbone in ['resnet18', 'resnet34', 'resnet50']
        if backbone == 'resnet18':
            self.net1 = resnet_modified(BasicBlock, [2, 2, 2, 2])
            self.net2 = resnet_modified(BasicBlock, [2, 2, 2, 2])
        elif backbone == 'resnet34':
            self.net1 = resnet_modified(BasicBlock, [3, 4, 6, 3])
            self.net2 = resnet_modified(BasicBlock, [3, 4, 6, 3])
        elif backbone == 'resnet50':
            self.net1 = resnet_modified(Bottleneck, [3, 4, 6, 3])
            self.net2 = resnet_modified(Bottleneck, [3, 4, 6, 3])
        else:
            raise
        self.backbone = backbone
        self.nchannels_dict = {'resnet18': {1: 64, 2: 128, 3: 256, 4: 512},
                               'resnet34': {1: 64, 2: 128, 3: 256, 4: 512},
                               'resnet50': {1: 256, 2: 512, 3: 1024, 4: 2048}}

        self.extra_1x1_conv_1 = nn.Conv2d(self.nchannels_dict[self.backbone][3], depth, kernel_size=1, stride=1,
                                          padding=0, bias=False)
        self.extra_1x1_conv_2 = nn.Conv2d(self.nchannels_dict[self.backbone][4], depth, kernel_size=1, stride=1,
                                          padding=0, bias=False)

        self.depth = depth
        self.nconvs = nconvs
        assert self.nconvs in [0, 1, 2, 3]

        if kernel_size == 3:
            padding = 1
        elif kernel_size == 1:
            padding = 0
        else:
            raise

        if self.nconvs in [1, 2, 3]:
            self.conv_combo_1 = nn.Sequential(nn.Conv2d(
                depth * 2, nfilters, kernel_size=kernel_size, padding=padding, stride=1), nn.BatchNorm2d(nfilters),
                nn.ReLU(inplace=True))

        if self.nconvs in [2, 3]:
            self.conv_combo_2 = nn.Sequential(nn.Conv2d(
                nfilters, nfilters, kernel_size=kernel_size, padding=padding, stride=1), nn.BatchNorm2d(nfilters),
                nn.ReLU(inplace=True))

        if self.nconvs == 3:
            self.conv_combo_3 = nn.Sequential(nn.Conv2d(
                nfilters, nfilters, kernel_size=kernel_size, padding=padding, stride=1), nn.BatchNorm2d(nfilters),
                nn.ReLU(inplace=True))

        if self.nconvs == 0:
            self.fc = nn.Linear(depth * 2, 2)
        else:
            self.fc = nn.Linear(nfilters, 2)

    def forward(self, img1, img2):

        _, _, fm19_1, fm10_1 = self.net1(img1)
        _, _, fm19_2, fm10_2 = self.net2(img2)

        # change the number of channels
        fm19_1 = self.extra_1x1_conv_1(fm19_1)  # batch_size * depth * 19 * 19
        fm19_2 = self.extra_1x1_conv_1(fm19_2)  # batch_size * depth * 19 * 19

        fm10_1 = self.extra_1x1_conv_2(fm10_1)  # batch_size * depth * 10 * 10
        fm10_2 = self.extra_1x1_conv_2(fm10_2)  # batch_size * depth * 10 * 10

        # depth-wise cross correlation
        sm19 = XCorr_depthwise(fm19_1, fm19_2, padding=9)  # batch_size * depth * 19 * 19
        sm10 = XCorr_depthwise(fm10_1, fm10_2, padding=5)  # batch_size * depth * 11 * 11

        # aggregate along the channel
        sm19_downsampled = F.adaptive_avg_pool2d(sm19, (sm10.size()[2], sm10.size()[3]))  # batch_size * depth * 11 * 11
        out = torch.cat([sm19_downsampled, sm10], 1)  # batch_size * (2*depth) * 11 * 11
        out = F.relu(out)

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


class psn_depthwise_cc_layerwise_3layers_l234(nn.Module):
    """
    Depthwise cross-correlation, layerwise aggregation (aggregate similarity maps from 3 layers).
    """
    def __init__(self, backbone='resnet18', nconvs=2, depth=128, nfilters=256, kernel_size=3):
        super(psn_depthwise_cc_layerwise_3layers_l234, self).__init__()
        assert backbone in ['resnet18', 'resnet34', 'resnet50']
        if backbone == 'resnet18':
            self.net1 = resnet_modified(BasicBlock, [2, 2, 2, 2])
            self.net2 = resnet_modified(BasicBlock, [2, 2, 2, 2])
        elif backbone == 'resnet34':
            self.net1 = resnet_modified(BasicBlock, [3, 4, 6, 3])
            self.net2 = resnet_modified(BasicBlock, [3, 4, 6, 3])
        elif backbone == 'resnet50':
            self.net1 = resnet_modified(Bottleneck, [3, 4, 6, 3])
            self.net2 = resnet_modified(Bottleneck, [3, 4, 6, 3])
        else:
            raise
        self.backbone = backbone
        self.nchannels_dict = {'resnet18': {1: 64, 2: 128, 3: 256, 4: 512},
                               'resnet34': {1: 64, 2: 128, 3: 256, 4: 512},
                               'resnet50': {1: 256, 2: 512, 3: 1024, 4: 2048}}

        self.extra_1x1_conv_1 = nn.Conv2d(self.nchannels_dict[self.backbone][2], depth, kernel_size=1, stride=1, padding=0, bias=False)
        self.extra_1x1_conv_2 = nn.Conv2d(self.nchannels_dict[self.backbone][3], depth, kernel_size=1, stride=1, padding=0, bias=False)
        self.extra_1x1_conv_3 = nn.Conv2d(self.nchannels_dict[self.backbone][4], depth, kernel_size=1, stride=1, padding=0, bias=False)

        self.depth = depth
        self.nconvs = nconvs
        assert self.nconvs in [0, 1, 2, 3]

        if kernel_size == 3:
            padding = 1
        elif kernel_size == 1:
            padding = 0
        else:
            raise

        if self.nconvs in [1, 2, 3]:
            self.conv_combo_1 = nn.Sequential(nn.Conv2d(
                depth*3, nfilters, kernel_size=kernel_size, padding=padding, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs in [2, 3]:
            self.conv_combo_2 = nn.Sequential(nn.Conv2d(
                nfilters, nfilters, kernel_size=kernel_size, padding=padding, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs == 3:
            self.conv_combo_3 = nn.Sequential(nn.Conv2d(
                nfilters, nfilters, kernel_size=kernel_size, padding=padding, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs == 0:
            self.fc = nn.Linear(depth*3, 2)
        else:
            self.fc = nn.Linear(nfilters, 2)

    def forward(self, img1, img2):

        _, fm38_1, fm19_1, fm10_1 = self.net1(img1)
        _, fm38_2, fm19_2, fm10_2 = self.net2(img2)

        # change the number of channels
        fm38_1 = self.extra_1x1_conv_1(fm38_1)  # batch_size * depth * 39 * 39
        fm38_2 = self.extra_1x1_conv_1(fm38_2)  # batch_size * depth * 39 * 39

        fm19_1 = self.extra_1x1_conv_2(fm19_1)  # batch_size * depth * 19 * 19
        fm19_2 = self.extra_1x1_conv_2(fm19_2)  # batch_size * depth * 19 * 19

        fm10_1 = self.extra_1x1_conv_3(fm10_1)  # batch_size * depth * 11 * 11
        fm10_2 = self.extra_1x1_conv_3(fm10_2)  # batch_size * depth * 11 * 11

        # depth-wise cross correlation
        sm38 = XCorr_depthwise(fm38_1, fm38_2, padding=19)  # batch_size * depth * 39 * 39
        sm19 = XCorr_depthwise(fm19_1, fm19_2, padding=9)  # batch_size * depth * 19 * 19
        sm10 = XCorr_depthwise(fm10_1, fm10_2, padding=5)   # batch_size * depth * 11 * 11

        # aggregate along the channel
        sm38_downsampled = F.adaptive_avg_pool2d(sm38, (sm10.size()[2], sm10.size()[3]))  # batch_size * depth * 11 * 11
        sm19_downsampled = F.adaptive_avg_pool2d(sm19, (sm10.size()[2], sm10.size()[3]))  # batch_size * depth * 11 * 11
        out = torch.cat([sm38_downsampled, sm19_downsampled, sm10], 1)  # batch_size * (3*depth) * 11 * 11
        out = F.relu(out)

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
