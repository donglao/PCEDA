import logging
import os
import os.path
from collections import deque
import itertools
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from UNet_CPN import UNet_CPN
import random

from PIL import Image
from torch.autograd import Variable

from cycada.data.adda_datasets import AddaDataLoader
from cycada.util import config_logging
from cycada.tools.util import make_variable
import matplotlib.pyplot as plt

def supervised_loss(score, label, weights=None):
    loss_fn_ = torch.nn.NLLLoss(weight=weights, size_average=True,
                                ignore_index=255)
    loss = loss_fn_(F.log_softmax(score, dim=1), label)
    return loss

def shuffle_label(label, num_classes):
    a,_,_ = label.shape
    for i in range(a):
        label[i,:,:] = random_permute(label[i,:,:], num_classes)

    return label

def random_permute(label, num_classes):
    ordering = torch.from_numpy(np.random.permutation(19)).long() # 19 classes 
    label2 = label.clone() # to fix bug
    for i in range(19):
        label2[label==i] = ordering[i]

    return label2

def one_hot(label, num_classes):
    a,b,c = label.shape
    label_new = torch.cuda.FloatTensor(a, num_classes, b, c).fill_(0)
    for i in range(a):
        for j in range(num_classes):
            label_new[i,j:,:] = (label[i,:,:]==j)

    return label_new


def train_cpn(dataset, datadir, lr, max_iter, crop_size, batch, output, checkpoint):
    # So data is sampled in consistent way
    loss_rec= deque(maxlen=100)
    net = UNet_CPN()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ])
    loader = AddaDataLoader(transform, dataset, datadir, downscale=None,
                            crop_size=crop_size, half_crop=None,
                            batch_size=batch, shuffle=True, num_workers=2)
    print('dataset', dataset)

    opt_rep = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)
    iteration = 0
    if checkpoint > 0:
        iteration = checkpoint
        net.load_state_dict(torch.load(output + '/' + str(iteration) + '.pth'))

    print('max iter:', max_iter)
    #net.load_state_dict(torch.load('120080.pth'))
    net.train()
    net.cuda()
    while iteration < max_iter:
        print(iteration)
        for im_s, im_t, label_s, label_t in loader:
            if iteration > max_iter:
                break
            ###########################
            # Randomly Permute Ground Truth Labels #
            ###########################
            im_s = make_variable(im_s, requires_grad=False)
            label_s = shuffle_label(make_variable(label_s, requires_grad=False), 19)

            # zero gradients for optimizer
            opt_rep.zero_grad()

            ###########################
            # Optimize CPN #
            ###########################

            score = net(im_s, one_hot(label_s, 19))

            loss = supervised_loss(score, label_s, weights=None)
            loss.backward()
            opt_rep.step()

            iteration += 1

            if iteration % 100 == 0:
                loss_rec.append(loss.item())
                print('Label reconstruction loss: ', np.mean(loss_rec), 'Iteration: ', iteration)

            ################
            # Save outputs #
            ################

            if iteration % 500 == 0:
                os.makedirs(output, exist_ok=True)
                torch.save(net.state_dict(), output + '/' + str(iteration) + '.pth')
                norm = plt.Normalize(vmin=0, vmax=255)
                image1 = np.argmax(score[0,:,:,:].cpu().detach().numpy(), axis=0)
                image2 = label_s[0, :, :].cpu().detach().numpy()
                image2[image2==255] =0
                plt.imsave(output + '/' + str(iteration) + '_r.png', norm(image1))
                plt.imsave(output + '/' + str(iteration) + '_l.png', norm(image2))

datadir = 'YOUR_FOLDER/cycada_release/data/'
dataset = ('cyclegta5', 'cyclegta5')
crop_size = 1024

lr = 0.0001
batch_size = 3
output = "CPN_output"
train_cpn(dataset, datadir, lr, 30000, crop_size, batch_size, output, 0)
train_cpn(dataset, datadir, lr/10, 60000, crop_size, batch_size, output, 30000)
train_cpn(dataset, datadir, lr/100, 100000, crop_size, batch_size, output, 60000)

