# -*-coding:utf-8 -*-
from __future__ import division, print_function

import os
from os.path import exists, join

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms, utils

import cv2
from utils.image_dataset import ImageFolderModified
from utils.inception_modified import InceptionSegmentation


class MkCam():
    def __init__(self):
        self.old_ckpt_path = '/home/ubuntu/projects/historical_solar/checkpoint/deepsolar_seg_pretrained.pth'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_size = 299
        self.batch_size = 1   # must be 1 for testing segmentation
        self.threshold = 0.5  # threshold probability to identify am image as positive
        self.level = 2
        self.transform_test = transforms.Compose([
                 transforms.Resize((self.input_size, self.input_size)),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                 ])
        self.model = InceptionSegmentation(num_outputs=2, level=self.level)
        self.model.load_existing_params(self.old_ckpt_path)
        self.model = self.model.to(self.device)

    def test_model(self, img_path):
        model = self.model
        img = Image.open(img_path).convert('RGB')
        img = self.transform_test(img)
        img = img.unsqueeze(0)
        inputs = img
        inputs = inputs.to(self.device)
        model.eval()
        CAM_list = []
        with torch.set_grad_enabled(False):
            _, outputs, CAM = model(inputs, testing=True)   # CAM is a 1 x 35 x 35 activation map
            prob = F.softmax(outputs, dim=1)
            preds = prob[:, 1] >= self.threshold

        CAM = CAM.squeeze(0).cpu().numpy()   # transform tensor into numpy array
        min_x = CAM.min()
        max_x = CAM.max()
        CAM = 255 * (CAM - min_x) / (max_x - min_x)
        cam_img = Image.fromarray(CAM)
        cam_img = cam_img.resize((700, 700))
        cam_img = cam_img.convert('L')
        return prob[:, 1], preds, cam_img

if __name__ == '__main__':
    
    cam_maker = MkCam()
    _, _, cam_img = cam_maker.test_model('D:/DeepSolar/coup/seq14_test/1334099/1334099_2017_0.png')
    cam_img.show()

