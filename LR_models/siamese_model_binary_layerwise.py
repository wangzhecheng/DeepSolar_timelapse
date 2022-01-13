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

        return x, fm38, fm19
        # return x256, fm19


class resnet18_modified_2(ResNet):
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
        # fm10 = self.layer4(fm19)
        # 512 x 10 x 10
        x256 = self.avgpool(fm19)
        x = x256.view(x256.size(0), -1)

        # x512 = self.avgpool(fm10)
        # x = x512.view(x512.size(0), -1)
        # x = self.fc(x)

        return x, fm75, fm38, fm19
        # return x256, fm19


class resnet18_modified_3(ResNet):
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


def XCross_depthwise(fm1, fm2, padding):
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


def XCross(fm1, fm2, padding):
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
    return torch.sum(out, keepdim=True, dim=1)  # batch_size x 1 x H x W


class sn_cc_layerwise(nn.Module):
    def __init__(self, nlayers=2, hidden=256):
        super(sn_cc_layerwise, self).__init__()
        self.net1 = resnet18_modified(BasicBlock, [2, 2, 2, 2])
        self.net1.fc = nn.Linear(256, 128)

        self.extra_1x1_conv = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)

        self.nlayers = nlayers
        assert self.nlayers in [1, 2, 3]

        if self.nlayers in [2, 3]:
            self.fc_combo_1 = nn.Sequential(nn.Linear(722, hidden), nn.ReLU(inplace=True))

        if self.nlayers == 3:
            self.fc_combo_2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

        if self.nlayers == 1:
            self.fc_final = nn.Linear(722, 2)
        else:
            self.fc_final = nn.Linear(hidden, 2)

    def forward(self, img1, img2):
        img1 = torch.cat([img1, img1, img1], 1)
        img2 = torch.cat([img2, img2, img2], 1)

        x1, fm38_1, fm19_1 = self.net1(img1)
        x2, fm38_2, fm19_2 = self.net1(img2)

        # cross correlation
        sm38 = XCross(fm38_1, fm38_2, padding=19)  # batch_size * 1 * 39 * 39
        sm19 = XCross(fm19_1, fm19_2, padding=9)   # batch_size * 1 * 19 * 19

        # aggregate
        sm38 = F.adaptive_avg_pool2d(sm38, (sm19.size()[2], sm19.size()[3]))  # batch_size * 128 * 19 * 19
        sm38 = sm38.view(sm38.size()[0], -1)  # batch_size * 361
        sm19 = sm19.view(sm38.size()[0], -1)  # batch_size * 361
        out = torch.cat([sm38, sm19], dim=1)  # batch_size * 722
        out = F.relu(out)

        # fc layers
        if self.nlayers in [2, 3]:
            out = self.fc_combo_1(out)
        if self.nlayers == 3:
            out = self.fc_combo_2(out)
        out = self.fc_final(out)
        return out


class sn_depthwise_cc_layerwise(nn.Module):
    def __init__(self, nconvs=2, nfilters=256):
        super(sn_depthwise_cc_layerwise, self).__init__()
        self.net1 = resnet18_modified(BasicBlock, [2, 2, 2, 2])
        self.net1.fc = nn.Linear(256, 128)

        self.extra_1x1_conv = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)

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

        x1, fm38_1, fm19_1 = self.net1(img1)
        x2, fm38_2, fm19_2 = self.net1(img2)

        # change the number of channels
        fm19_1 = self.extra_1x1_conv(fm19_1) # batch_size * 128 * 19 * 19
        fm19_2 = self.extra_1x1_conv(fm19_2) # batch_size * 128 * 19 * 19

        # depth-wise cross correlation
        sm38 = XCross_depthwise(fm38_1, fm38_2, padding=19)  # batch_size * 128 * 38 * 38
        sm19 = XCross_depthwise(fm19_1, fm19_2, padding=9)   # batch_size * 128 * 19 * 19

        # aggregate along the channel
        sm38_downsampled = F.adaptive_avg_pool2d(sm38, (sm19.size()[2], sm19.size()[3]))  # batch_size * 128 * 19 * 19
        out = torch.cat([sm38_downsampled, sm19], 1)  # batch_size * 256 * 19 * 19
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


class sn_depthwise_cc_layerwise_3x3(nn.Module):
    def __init__(self, nconvs=2, nfilters=256):
        super(sn_depthwise_cc_layerwise_3x3, self).__init__()
        self.net1 = resnet18_modified(BasicBlock, [2, 2, 2, 2])
        self.net1.fc = nn.Linear(256, 128)

        self.extra_1x1_conv = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)

        self.nconvs = nconvs
        assert self.nconvs in [0, 1, 2, 3]

        if self.nconvs in [1, 2, 3]:
            self.conv_combo_1 = nn.Sequential(nn.Conv2d(
                256, nfilters, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs in [2, 3]:
            self.conv_combo_2 = nn.Sequential(nn.Conv2d(
                nfilters, nfilters, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs == 3:
            self.conv_combo_3 = nn.Sequential(nn.Conv2d(
                nfilters, nfilters, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs == 0:
            self.fc = nn.Linear(256, 2)
        else:
            self.fc = nn.Linear(nfilters, 2)

    def forward(self, img1, img2):
        img1 = torch.cat([img1, img1, img1], 1)
        img2 = torch.cat([img2, img2, img2], 1)

        x1, fm38_1, fm19_1 = self.net1(img1)
        x2, fm38_2, fm19_2 = self.net1(img2)

        # change the number of channels
        fm19_1 = self.extra_1x1_conv(fm19_1) # batch_size * 128 * 19 * 19
        fm19_2 = self.extra_1x1_conv(fm19_2) # batch_size * 128 * 19 * 19

        # depth-wise cross correlation
        sm38 = XCross_depthwise(fm38_1, fm38_2, padding=19)  # batch_size * 128 * 38 * 38
        sm19 = XCross_depthwise(fm19_1, fm19_2, padding=9)   # batch_size * 128 * 19 * 19

        # aggregate along the channel
        sm38_downsampled = F.adaptive_avg_pool2d(sm38, (sm19.size()[2], sm19.size()[3]))  # batch_size * 128 * 19 * 19
        out = torch.cat([sm38_downsampled, sm19], 1)  # batch_size * 256 * 19 * 19
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



class sn_depthwise_cc_layerwise_3x3_3layers(nn.Module):
    """
    Depthwise cross-correlation, layerwise aggregation (aggregate similarity maps from 3 layers), with 3x3 convolution kernel.
    """
    def __init__(self, nconvs=2, nfilters=256):
        super(sn_depthwise_cc_layerwise_3x3_3layers, self).__init__()
        self.net1 = resnet18_modified_2(BasicBlock, [2, 2, 2, 2])
        self.net1.fc = nn.Linear(256, 128)

        self.extra_1x1_conv_1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.extra_1x1_conv_3 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)

        self.nconvs = nconvs
        assert self.nconvs in [0, 1, 2, 3]

        if self.nconvs in [1, 2, 3]:
            self.conv_combo_1 = nn.Sequential(nn.Conv2d(
                384, nfilters, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs in [2, 3]:
            self.conv_combo_2 = nn.Sequential(nn.Conv2d(
                nfilters, nfilters, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs == 3:
            self.conv_combo_3 = nn.Sequential(nn.Conv2d(
                nfilters, nfilters, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs == 0:
            self.fc = nn.Linear(384, 2)
        else:
            self.fc = nn.Linear(nfilters, 2)

    def forward(self, img1, img2):
        img1 = torch.cat([img1, img1, img1], 1)
        img2 = torch.cat([img2, img2, img2], 1)

        x1, fm75_1, fm38_1, fm19_1 = self.net1(img1)
        x2, fm75_2, fm38_2, fm19_2 = self.net1(img2)

        # change the number of channels
        fm75_1 = self.extra_1x1_conv_1(fm75_1)  # batch_size * 128 * 75 * 75
        fm75_2 = self.extra_1x1_conv_1(fm75_2)  # batch_size * 128 * 75 * 75

        fm19_1 = self.extra_1x1_conv_3(fm19_1)  # batch_size * 128 * 19 * 19
        fm19_2 = self.extra_1x1_conv_3(fm19_2)  # batch_size * 128 * 19 * 19

        # depth-wise cross correlation
        sm75 = XCross_depthwise(fm75_1, fm75_2, padding=37)  # batch_size * 128 * 75 * 75
        sm38 = XCross_depthwise(fm38_1, fm38_2, padding=19)  # batch_size * 128 * 39 * 39
        sm19 = XCross_depthwise(fm19_1, fm19_2, padding=9)   # batch_size * 128 * 19 * 19

        # aggregate along the channel
        sm75_downsampled = F.adaptive_avg_pool2d(sm75, (sm19.size()[2], sm19.size()[3]))  # batch_size * 128 * 19 * 19
        sm38_downsampled = F.adaptive_avg_pool2d(sm38, (sm19.size()[2], sm19.size()[3]))  # batch_size * 128 * 19 * 19
        out = torch.cat([sm75_downsampled, sm38_downsampled, sm19], 1)  # batch_size * 384 * 19 * 19
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


class sn_depthwise_cc_layerwise_last2(nn.Module):
    def __init__(self, nconvs=2, nfilters=256):
        super(sn_depthwise_cc_layerwise_last2, self).__init__()
        self.net1 = resnet18_modified_3(BasicBlock, [2, 2, 2, 2])
        self.net1.fc = nn.Linear(256, 128)

        self.extra_1x1_conv = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)

        self.nconvs = nconvs
        assert self.nconvs in [0, 1, 2, 3]

        if self.nconvs in [1, 2, 3]:
            self.conv_combo_1 = nn.Sequential(nn.Conv2d(
                512, nfilters, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs in [2, 3]:
            self.conv_combo_2 = nn.Sequential(nn.Conv2d(
                nfilters, nfilters, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs == 3:
            self.conv_combo_3 = nn.Sequential(nn.Conv2d(
                nfilters, nfilters, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs == 0:
            self.fc = nn.Linear(512, 2)
        else:
            self.fc = nn.Linear(nfilters, 2)

    def forward(self, img1, img2):
        img1 = torch.cat([img1, img1, img1], 1)
        img2 = torch.cat([img2, img2, img2], 1)

        _, _, fm19_1, fm10_1 = self.net1(img1)
        _, _, fm19_2, fm10_2 = self.net1(img2)

        # change the number of channels
        fm10_1 = self.extra_1x1_conv(fm10_1)  # batch_size * 256 * 10 * 10
        fm10_2 = self.extra_1x1_conv(fm10_2)  # batch_size * 256 * 10 * 10

        # depth-wise cross correlation
        sm19 = XCross_depthwise(fm19_1, fm19_2, padding=9)   # batch_size * 256 * 19 * 19
        sm10 = XCross_depthwise(fm10_1, fm10_2, padding=5)  # batch_size * 256 * 11 * 11

        # aggregate along the channel
        sm19_downsampled = F.adaptive_avg_pool2d(sm19, (sm10.size()[2], sm10.size()[3]))  # batch_size * 256 * 11 * 11
        out = torch.cat([sm19_downsampled, sm10], 1)  # batch_size * 512 * 11 * 11
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


class sn_depthwise_cc_layerwise_last3(nn.Module):
    """
    Depthwise cross-correlation, layerwise aggregation (aggregate similarity maps from 3 layers), with 3x3 convolution kernel.
    """
    def __init__(self, nconvs=2, nfilters=256):
        super(sn_depthwise_cc_layerwise_last3, self).__init__()
        self.net1 = resnet18_modified_3(BasicBlock, [2, 2, 2, 2])
        self.net1.fc = nn.Linear(256, 128)

        self.extra_1x1_conv_2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.extra_1x1_conv_3 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)

        self.nconvs = nconvs
        assert self.nconvs in [0, 1, 2, 3]

        if self.nconvs in [1, 2, 3]:
            self.conv_combo_1 = nn.Sequential(nn.Conv2d(
                384, nfilters, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs in [2, 3]:
            self.conv_combo_2 = nn.Sequential(nn.Conv2d(
                nfilters, nfilters, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs == 3:
            self.conv_combo_3 = nn.Sequential(nn.Conv2d(
                nfilters, nfilters, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(nfilters), nn.ReLU(inplace=True))

        if self.nconvs == 0:
            self.fc = nn.Linear(384, 2)
        else:
            self.fc = nn.Linear(nfilters, 2)

    def forward(self, img1, img2):
        img1 = torch.cat([img1, img1, img1], 1)
        img2 = torch.cat([img2, img2, img2], 1)

        _, fm38_1, fm19_1, fm10_1 = self.net1(img1)
        _, fm38_2, fm19_2, fm10_2 = self.net1(img2)

        # change the number of channels
        fm19_1 = self.extra_1x1_conv_2(fm19_1)  # batch_size * 128 * 19 * 19
        fm19_2 = self.extra_1x1_conv_2(fm19_2)  # batch_size * 128 * 19 * 19

        fm10_1 = self.extra_1x1_conv_3(fm10_1)  # batch_size * 128 * 10 * 10
        fm10_2 = self.extra_1x1_conv_3(fm10_2)  # batch_size * 128 * 10 * 10

        # depth-wise cross correlation
        sm38 = XCross_depthwise(fm38_1, fm38_2, padding=19)  # batch_size * 128 * 39 * 39
        sm19 = XCross_depthwise(fm19_1, fm19_2, padding=9)   # batch_size * 128 * 19 * 19
        sm10 = XCross_depthwise(fm10_1, fm10_2, padding=5)  # batch_size * 128 * 11 * 11


        # aggregate along the channel
        sm38_downsampled = F.adaptive_avg_pool2d(sm38, (sm10.size()[2], sm10.size()[3]))  # batch_size * 128 * 11 * 11
        sm19_downsampled = F.adaptive_avg_pool2d(sm19, (sm10.size()[2], sm10.size()[3]))  # batch_size * 128 * 11 * 11
        out = torch.cat([sm38_downsampled, sm19_downsampled, sm10], 1)  # batch_size * 384 * 11 * 11
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