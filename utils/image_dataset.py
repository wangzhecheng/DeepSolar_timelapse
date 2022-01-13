from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms, utils
import torchvision.transforms.functional as TF

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


class ImageFolderModified(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.idx2dir = []
        self.path_list = []
        for subdir in sorted(os.listdir(self.root_dir)):
            if not os.path.isfile(subdir):
                self.idx2dir.append(subdir)
        for class_idx, subdir in enumerate(self.idx2dir):
            class_dir = os.path.join(self.root_dir, subdir)
            for f in os.listdir(class_dir):
                if f[-4:] in ['.png', '.jpg', 'JPEG', 'jpeg']:
                    self.path_list.append([os.path.join(class_dir, f), class_idx])

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        img_path, class_idx = self.path_list[idx]
        image = Image.open(img_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        sample = [image, class_idx, img_path]
        return sample


class BinaryImageFolder(Dataset):
    def __init__(self, root_dirs, transform):
        """
        :param root_dirs: the list of root directories, the subdirectory of each root directory must be '0' and '1'
        :param transform: pytorch transform functions
        """
        self.root_dirs = root_dirs
        self.transform = transform
        self.path_list = []
        for root_dir in self.root_dirs:
            assert exists(join(root_dir, '0')) and exists(join(root_dir, '1'))
            for class_idx in [0, 1]:
                class_dir = join(root_dir, str(class_idx))
                for f in os.listdir(class_dir):
                    if f[-4:] in ['.png', '.jpg', 'JPEG', 'jpeg']:
                        self.path_list.append([join(class_dir, f), class_idx])

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        img_path, class_idx = self.path_list[idx]
        image = Image.open(img_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        sample = [image, class_idx, img_path]
        return sample


class ImagePairDataset(Dataset):
    """
    :param root_dirs: the list of root directories, the subdirectory of each root directory must be '0' and '1'
    :param reference_mapping_paths: the list of path to reference_mapping (dict)
    :param is_train: boolean indicating whether it is a training set
    :param binary: boolean indicating whether the images are binary
    :param transform: pytorch transform functions
    """
    def __init__(self, root_dirs, reference_mapping_paths, is_train, binary, transform):
        self.couple_list = []
        self.is_train = is_train
        self.binary = binary
        self.transform = transform

        assert len(root_dirs) == len(reference_mapping_paths)
        for i, root_dir in enumerate(root_dirs):
            reference_mapping_path = reference_mapping_paths[i]
            with open(reference_mapping_path, 'rb') as f:
                reference_mapping = pickle.load(f)

            x = root_dir.split('/')[-1]
            assert x in ['train', 'val', 'test']

            for class_idx in [0, 1]:
                class_dir = join(root_dir, str(class_idx))
                for f in os.listdir(class_dir):
                    if f[-4:] in ['.png', '.jpg', 'JPEG', 'jpeg']:
                        target_path = join(class_dir, f)
                        target_subpath = join(x, str(class_idx), f)
                        if target_subpath in reference_mapping:
                            for ref_subpath in reference_mapping[target_subpath]:
                                ref_path = root_dir.replace('/' + x, '/' + ref_subpath)
                                if exists(ref_path):
                                    self.couple_list.append((ref_path, target_path, class_idx))

    def __len__(self):
        return len(self.couple_list)

    def __getitem__(self, index):
        each_couple_list = self.couple_list[index]
        img_ref = Image.open(each_couple_list[0])
        img_tar = Image.open(each_couple_list[1])
        if not self.binary:
            if not img_ref.mode == 'RGB':
                img_ref = img_ref.convert('RGB')
            if not img_tar.mode == 'RGB':
                img_tar = img_tar.convert('RGB')

        if self.is_train:
            angle = random.choice([0, 90, 180, 270])
            img_ref = TF.rotate(img_ref, angle)
            img_tar = TF.rotate(img_tar, angle)

        img_ref = self.transform(img_ref)
        img_tar = self.transform(img_tar)

        label = each_couple_list[2]
        return img_ref, img_tar, label
        # return img_tar, label


class SequenceDataset(Dataset):
    def __init__(self, folder_dir, transform):
        """
        :param folder_dir: e.g. "/home/ubuntu/projects/data/deepsolar2/cleaned/sequence_0/6083_128565_2012"
        """
        self.path_list = []
        self.transform = transform
        for f in sorted(os.listdir(folder_dir)):
            if f[-4:] in ['.png', '.jpg', 'JPEG', 'jpeg']:
                self.path_list.append(join(folder_dir, f))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        img_path = self.path_list[idx]
        image = Image.open(img_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        sample = [image, img_path.split('/')[-1]]
        return sample


class FolderDirsDataset(Dataset):
    def __init__(self, dirs_list, transform):
        """
        :param dirs_list: list. Length: number of classes. Each entries is a list of directories belonging to its class.
        """
        self.sample_list = []
        self.transform = transform
        for i, dirs in enumerate(dirs_list):
            for dir in dirs:
                for f in os.listdir(dir):
                    if f[-4:] in ['.png', '.jpg', 'JPEG', 'jpeg']:
                        self.sample_list.append((join(dir, f), i))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_path, class_idx = self.sample_list[idx]
        image = Image.open(img_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        sample = [image, class_idx]
        return sample


class FolderDirsDatasetMultiLabels(Dataset):
    def __init__(self, dirs_list, transform):
        """
        :param dirs_list: list. Length: number of classes. Each entries is a list of directories belonging to its class.
        """
        self.sample_list = []
        self.transform = transform
        for i, dirs in enumerate(dirs_list):
            for dir in dirs:
                for f in os.listdir(dir):
                    if f[-4:] in ['.png', '.jpg', 'JPEG', 'jpeg']:
                        if i == 0:
                            self.sample_list.append((join(dir, f), 0, 0))
                        elif i == 1:
                            self.sample_list.append((join(dir, f), 1, 0))
                        else:
                            self.sample_list.append((join(dir, f), 1, 1))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_path, class_idx_1, class_idx_2 = self.sample_list[idx]
        image = Image.open(img_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        sample = [image, torch.tensor([class_idx_1, class_idx_2], dtype=torch.float)]
        return sample


class FolderDirsDatasetMultiLabelsForSolarTypes(Dataset):
    def __init__(self, dirs_list, transform):
        """
        :param dirs_list: list. Length: number of classes. Each entries is a list of directories belonging to its class.
        """
        self.sample_list = []
        self.transform = transform
        for i, dirs in enumerate(dirs_list):
            for dir in dirs:
                for f in os.listdir(dir):
                    if f[-4:] in ['.png', '.jpg', 'JPEG', 'jpeg']:
                        if i == 0:
                            self.sample_list.append((join(dir, f), [0, 0, 0, 0]))  # negative
                        elif i == 1:
                            self.sample_list.append((join(dir, f), [1, 0, 0, 0]))  # solar water heating
                        elif i == 2:
                            self.sample_list.append((join(dir, f), [1, 1, 0, 0]))   # residential solar
                        elif i == 3:
                            self.sample_list.append((join(dir, f), [1, 1, 1, 0]))   # commercial solar
                        else:
                            self.sample_list.append((join(dir, f), [1, 1, 1, 1]))  # utility-scale solar

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_path, class_labels = self.sample_list[idx]
#         print(img_path)
#         class_idx_1, class_idx_2, class_idx_3, class_idx_4 = class_labels
        image = Image.open(img_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        sample = [image, torch.tensor(class_labels, dtype=torch.float)]
        return sample
