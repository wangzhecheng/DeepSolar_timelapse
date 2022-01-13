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
# from LR_models.couple_model import *
# from LR_models.siamese_model_binary import *
from LR_models.siamese_model_binary_layerwise import *

# Configuration
# directory for loading training/validation/test data
data_dirs_dict = {
    'train': ['/home/ubuntu/projects/data/deepsolar2/cleaned/LR_0_binary/train',
              '/home/ubuntu/projects/data/deepsolar2/cleaned/LR_1_binary/train',
              '/home/ubuntu/projects/data/deepsolar2/cleaned/LR_2_binary/train'],
    'val': ['/home/ubuntu/projects/data/deepsolar2/cleaned/LR_0_binary/val',
            '/home/ubuntu/projects/data/deepsolar2/cleaned/LR_1_binary/val',
            '/home/ubuntu/projects/data/deepsolar2/cleaned/LR_2_binary/val'],
    'test': ['/home/ubuntu/projects/data/deepsolar2/cleaned/LR_0_binary/test',
             '/home/ubuntu/projects/data/deepsolar2/cleaned/LR_1_binary/test',
             '/home/ubuntu/projects/data/deepsolar2/cleaned/LR_2_binary/test'],
}

reference_mapping_paths_dict = {
    'train': ['/home/ubuntu/projects/data_preparation/reference_mapping/reference_mapping_LR_0.pickle',
              '/home/ubuntu/projects/data_preparation/reference_mapping/reference_mapping_LR_1_single.pickle',
              '/home/ubuntu/projects/data_preparation/reference_mapping/reference_mapping_LR_2_single.pickle'],
    'val': ['/home/ubuntu/projects/data_preparation/reference_mapping/reference_mapping_LR_0.pickle',
            '/home/ubuntu/projects/data_preparation/reference_mapping/reference_mapping_LR_1_single.pickle',
            '/home/ubuntu/projects/data_preparation/reference_mapping/reference_mapping_LR_2_single.pickle'],
    'test': ['/home/ubuntu/projects/data_preparation/reference_mapping/reference_mapping_LR_0.pickle',
             '/home/ubuntu/projects/data_preparation/reference_mapping/reference_mapping_LR_1_single.pickle',
             '/home/ubuntu/projects/data_preparation/reference_mapping/reference_mapping_LR_2_single.pickle']
}

# path to load old model/checkpoint, "None" if not loading.
# old_ckpt_path = None
old_ckpt_path = '/home/ubuntu/projects/DeepSolarII/checkpoint/resnet18-5c106cde.pth'
# directory for saving model/checkpoint
ckpt_save_dir = 'checkpoint/LR/LR_012_dwcc_3x3_layerwise_last3'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainable_params = None     # layers or modules set to be trainable. "None" if training all layers
model_name = 'LR'  # the prefix of the filename for saving model/checkpoint
return_best = True           # whether to return the best model according to the validation metrics
if_early_stop = True         # whether to stop early after validation metrics doesn't improve for definite number of epochs
input_size = 299              # image size fed into the model
imbalance_rate = 1.0            # weight given to the positive (rarer) samples in loss function
# learning_rate = 0.01          # learning rate
weight_decay = 0           # l2 regularization coefficient
batch_size = 64
num_epochs = 100               # number of epochs to train
lr_decay_rate = 0.95           # learning rate decay rate for each decay step
lr_decay_epochs = 4          # number of epochs for one learning rate decay
early_stop_epochs = 10        # after validation metrics doesn't improve for "early_stop_epochs" epochs, stop the training.
save_epochs = 50              # save the model/checkpoint every "save_epochs" epochs
# threshold = 0.2               # threshold probability to identify am image as positive
nfilters = 256
lr_list = [0.01, 0.03]
# lr_decay_epochs_list = [4, 10]
nconvs_list = [2, 3]
nfilters_list = [512, 640, 768]
# weight_decay_list = [0, 0.001]
threshold_list = np.linspace(0.0, 1.0, 101).tolist()


def RandomRotationNew(image):
    angle = random.choice([0, 90, 180, 270])
    image = TF.rotate(image, angle)
    return image

def only_train(model, trainable_params):
    """trainable_params: The list of parameters and modules that are set to be trainable.
    Set require_grad = False for all those parameters not in the trainable_params"""
    print('Only the following layers:')
    for name, p in model.named_parameters():
        p.requires_grad = False
        for target in trainable_params:
            if target == name or target in name:
                p.requires_grad = True
                print('    ' + name)
                break

def metrics(stats):
    """stats: {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
    return: must be a single number """
    # precision = (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FP'] + 0.00001)
    # recall = (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FN'] + 0.00001)
    # F1 = 2.0 * stats['TP'] / (2 * stats['TP'] + stats['FP'] + stats['FN'])
    spec = (stats['TN'] + 0.00001) * 1.0 / (stats['TN'] + stats['FP'] + 0.00001)
    sens = (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FN'] + 0.00001)
    return 2.0 * spec * sens / (spec + sens + 1e-7)


def train_model(model, model_name, dataloaders, criterion, optimizer, metrics, num_epochs, training_log=None,
                verbose=True, return_best=True, if_early_stop=True, early_stop_epochs=10, scheduler=None,
                save_dir=None, save_epochs=5):
    since = time.time()
    if not training_log:
        training_log = dict()
        training_log['train_loss_history'] = []
        training_log['val_loss_history'] = []
        training_log['val_metric_value_history'] = []
        training_log['epoch_best_threshold_history'] = []
        training_log['current_epoch'] = -1
    current_epoch = training_log['current_epoch'] + 1

    best_model_wts = copy.deepcopy(model.state_dict())
    best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
    best_log = copy.deepcopy(training_log)

    best_metric_value = -np.inf
    best_threshold = 0
    nodecrease = 0  # to count the epochs that val loss doesn't decrease
    early_stop = False

    for epoch in range(current_epoch, current_epoch + num_epochs):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            stats = {x: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for x in threshold_list}

            # Iterate over data.
            for inputs_ref, inputs_tar, labels in tqdm(dataloaders[phase]):
                inputs_ref = inputs_ref.to(device)
                inputs_tar = inputs_tar.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    if phase == 'train':
                        outputs = model(inputs_ref, inputs_tar)
                        loss = criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()

                    else:
                        outputs = model(inputs_ref, inputs_tar)
                        loss = criterion(outputs, labels)
                        # val phase: calculate metrics under different threshold
                        prob = F.softmax(outputs, dim=1)
                        for threshold in threshold_list:
                            preds = prob[:, 1] >= threshold
                            stats[threshold]['TP'] += torch.sum((preds == 1) * (labels == 1)).cpu().item()
                            stats[threshold]['TN'] += torch.sum((preds == 0) * (labels == 0)).cpu().item()
                            stats[threshold]['FP'] += torch.sum((preds == 1) * (labels == 0)).cpu().item()
                            stats[threshold]['FN'] += torch.sum((preds == 0) * (labels == 1)).cpu().item()

                # loss accumulation
                running_loss += loss.item() * inputs_ref.size(0)

            training_log['current_epoch'] = epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            if phase == 'train':
                training_log['train_loss_history'].append(epoch_loss)
                if scheduler is not None:
                    scheduler.step()
                if verbose:
                    print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'val':
                epoch_best_threshold = 0.0
                epoch_max_metrics = 0.0
                for threshold in threshold_list:
                    metric_value = metrics(stats[threshold])
                    if metric_value > epoch_max_metrics:
                        epoch_best_threshold = threshold
                        epoch_max_metrics = metric_value
                spec = (stats[epoch_best_threshold]['TN'] + 0.00001) * 1.0 / (stats[epoch_best_threshold]['TN'] + stats[epoch_best_threshold]['FP'] + 0.00001)
                sens = (stats[epoch_best_threshold]['TP'] + 0.00001) * 1.0 / (stats[epoch_best_threshold]['TP'] + stats[epoch_best_threshold]['FN'] + 0.00001)

                if verbose:
                    print('{} Loss: {:.4f} Metrics: {:.4f} Threshold: {:.4f} Sensitivity: {:.4f} Specificity {:4f}'.format(phase, epoch_loss,
                                                                                     epoch_max_metrics, epoch_best_threshold, sens, spec))

                training_log['val_metric_value_history'].append(epoch_max_metrics)
                training_log['val_loss_history'].append(epoch_loss)
                training_log['epoch_best_threshold_history'].append(epoch_best_threshold)

                # deep copy the model
                if epoch_max_metrics > best_metric_value:
                    best_metric_value = epoch_max_metrics
                    best_threshold = epoch_best_threshold
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
                    best_log = copy.deepcopy(training_log)
                    nodecrease = 0
                else:
                    nodecrease += 1

            if nodecrease >= early_stop_epochs:
                early_stop = True

        if save_dir and epoch % save_epochs == 0 and epoch > 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_log': training_log
            }
            torch.save(checkpoint,
                       os.path.join(save_dir, model_name + '_' + str(training_log['current_epoch']) + '.tar'))

        if if_early_stop and early_stop:
            print('Early stopped!')
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation metric value: {:4f}'.format(best_metric_value))
    print('Best validation threshold: {:4f}'.format(best_threshold))

    # load best model weights
    if return_best:
        model.load_state_dict(best_model_wts)
        optimizer.load_state_dict(best_optimizer_wts)
        training_log = best_log

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_log': training_log
    }
    torch.save(checkpoint,
               os.path.join(save_dir, model_name + '_' + str(training_log['current_epoch']) + '_last.tar'))

    return model, training_log, best_metric_value, best_threshold


def test_model(model, dataloader, metrics, threshold_list):
    stats = {x: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for x in threshold_list}
    metric_values = {}
    model.eval()
    for inputs_ref, inputs_tar, labels in tqdm(dataloader):
        inputs_ref = inputs_ref.to(device)
        inputs_tar = inputs_tar.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs_ref, inputs_tar)
            prob = F.softmax(outputs, dim=1)
            for threshold in threshold_list:
                preds = prob[:, 1] >= threshold
                stats[threshold]['TP'] += torch.sum((preds == 1) * (labels == 1)).cpu().item()
                stats[threshold]['TN'] += torch.sum((preds == 0) * (labels == 0)).cpu().item()
                stats[threshold]['FP'] += torch.sum((preds == 1) * (labels == 0)).cpu().item()
                stats[threshold]['FN'] += torch.sum((preds == 0) * (labels == 1)).cpu().item()

    for threshold in threshold_list:
        metric_values[threshold] = metrics(stats[threshold])

    return stats, metric_values


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((input_size, input_size)),    # transfer to input size
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])
}


if __name__ == '__main__':
    # data
    image_datasets = {x: ImagePairDataset(data_dirs_dict[x],
                                          reference_mapping_paths_dict[x],
                                          is_train=(x == 'train'),
                                          binary=True,
                                          transform=data_transforms[x]) for x in ['train', 'val', 'test']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                       shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}

    print('Training set size: ' + str(len(image_datasets['train'])))
    print('Validation set size: ' + str(len(image_datasets['val'])))
    print('Test set size: ' + str(len(image_datasets['test'])))

    results_dict = {x: {y: {z: {} for z in nfilters_list} for y in lr_list} for x in nconvs_list}

    if not os.path.exists(ckpt_save_dir):
        os.mkdir(ckpt_save_dir)

    # model
    for nconvs in nconvs_list:
        for learning_rate in lr_list:
            for nfilters in nfilters_list:
                print('----------------------- ' +
                      str(nconvs) + ', ' +
                      str(learning_rate) + ', ' +
                      str(nfilters) +
                      ' -----------------------')
                model = sn_depthwise_cc_layerwise_last3(nconvs=nconvs, nfilters=nfilters)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                       weight_decay=weight_decay, amsgrad=True)
                class_weight = torch.tensor([1, imbalance_rate], dtype=torch.float).cuda()
                loss_fn = nn.CrossEntropyLoss(weight=class_weight)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_epochs, gamma=lr_decay_rate)

                # load old parameters
                if old_ckpt_path:
                    checkpoint = torch.load(old_ckpt_path)
                    if old_ckpt_path[-4:] == '.tar':  # it is a checkpoint dictionary rather than just model parameters
                        model.load_state_dict(checkpoint['model_state_dict'])
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        training_log = checkpoint['training_log']
                    else:
                        del checkpoint['fc.weight']
                        del checkpoint['fc.bias']
                        model.net1.load_state_dict(checkpoint, strict=False)
                        # model.net2.load_state_dict(checkpoint, strict=False)
                        training_log = None  # start from scratch
                    print('Old checkpoint loaded: ' + old_ckpt_path)

                model = model.to(device)

                # fix some layers and make others trainable
                if trainable_params:
                    only_train(model, trainable_params)

                best_model, _, best_metric_value, best_threshold = train_model(model, model_name=model_name + '_nconvs_' + str(
                    nconvs) + '_lr_' + str(learning_rate) + '_nfilters_' + str(nfilters),
                                                                      dataloaders=dataloaders_dict, criterion=loss_fn,
                                                                      optimizer=optimizer, metrics=metrics,
                                                                      num_epochs=num_epochs,
                                                                      training_log=training_log, verbose=True,
                                                                      return_best=return_best,
                                                                      if_early_stop=if_early_stop,
                                                                      early_stop_epochs=early_stop_epochs,
                                                                      scheduler=scheduler, save_dir=ckpt_save_dir,
                                                                      save_epochs=save_epochs)

                print('Begin test ...')
                test_stats, test_metric_values = test_model(best_model, dataloaders_dict['test'], metrics, threshold_list=threshold_list)

                best_threshold_test, best_metric_value_test = sorted(list(test_metric_values.items()), key=lambda tup: tup[1], reverse=True)[0]

                results_dict[nconvs][learning_rate][nfilters] = {'best_metrics_val': best_metric_value,
                                                                 'best_threshold_val': best_threshold,
                                                                 'best_metric_test': best_metric_value_test,
                                                                 'best_threshold_test': best_threshold_test,
                                                                 'test_metrics_with_val_best_threshold':
                                                                     test_metric_values[best_threshold],
                                                                 }

                with open(join(ckpt_save_dir, 'results_dict.pickle'), 'wb') as f:
                    pickle.dump(results_dict, f)

    # for nconvs in nconvs_list:
    #     for learning_rate in lr_list:
    #         for nfilters in nfilters_list:
    #             print('nconvs: ', nconvs, 'learning_rate: ', learning_rate,
    #                   'nfilters: ', nfilters,
    #                 'metric: ', results_dict[nconvs][learning_rate][nfilters]['metrics'],
    #                 'threshold: ', results_dict[nconvs][learning_rate][nfilters]['threshold'])
