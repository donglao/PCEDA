"""
UNet-CPN
"""
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from skimage.transform import resize
from torch.utils.data import DataLoader
import torchvision

# Utility Functions
''' when filter kernel= 3x3, padding=1 makes in&out matrix same size'''
def conv_bn_leru(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
    )

def down_pooling():
    return nn.MaxPool2d(2)

def up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet_CPN(nn.Module):
    def __init__(self):
        super().__init__()
        # Encode img
        self.conv1 = conv_bn_leru(3,16)
        self.conv2 = conv_bn_leru(16, 32)
        self.conv3 = conv_bn_leru(32, 64)
        self.conv4 = conv_bn_leru(64, 128)
        self.conv5 = conv_bn_leru(128, 256)
        self.conv_added = conv_bn_leru(256, 256)
        self.down_pooling = nn.MaxPool2d(2)

        # Encode seg map
        self.conv_seg_1 = conv_bn_leru(19,16)
        self.conv_seg_2 = conv_bn_leru(16, 32)
        self.conv_seg_3 = conv_bn_leru(32, 64)
        self.conv_seg_4 = conv_bn_leru(64, 128)
        self.conv_seg_5 = conv_bn_leru(128, 256)
        self.conv_seg_added = conv_bn_leru(256, 256)
        self.down_pooling_seg = nn.MaxPool2d(2)

        # Decode seg map
        self.up_pool_added = up_pooling(512, 256)
        self.conv_added_2 = conv_bn_leru(512, 256)
        self.up_pool6 = up_pooling(256, 128)
        self.conv6 = conv_bn_leru(256, 128)
        self.up_pool7 = up_pooling(128, 64)
        self.conv7 = conv_bn_leru(128, 64)
        self.up_pool8 = up_pooling(64, 32)
        self.conv8 = conv_bn_leru(64, 32)
        self.up_pool9 = up_pooling(32, 16)
        self.conv9 = conv_bn_leru(32, 16)

        # fully connected layer, 19 classes
        self.fc = nn.Conv2d(32, 19, kernel_size=1,
                            stride=1, padding=0, bias=True)


    def forward(self,x,y):
        _, _, h, w = y.size()
        # Encode img
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)
        p5 = self.down_pooling(x5)
        x_added = self.conv_added(p5)

        # Encode seg mask

        y1 = self.conv_seg_1(y)
        pp1 = self.down_pooling_seg(y1)
        y2 = self.conv_seg_2(pp1)
        pp2 = self.down_pooling_seg(y2)
        y3 = self.conv_seg_3(pp2)
        pp3 = self.down_pooling_seg(y3)
        y4 = self.conv_seg_4(pp3)
        pp4 = self.down_pooling_seg(y4)
        y5 = self.conv_seg_5(pp4)
        pp5 = self.down_pooling_seg(y5)
        y_added = self.conv_seg_added(pp5)

        # combination in the bottleneck
        combination = torch.cat([x_added, y_added], dim=1)

        # Decode reconstructed seg map
        p_added = self.up_pool_added(combination)
        x_added2 = torch.cat([p_added, x5], dim=1)
        x_added2 = self.conv_added_2(x_added2)
        p6 = self.up_pool6(x_added2)
        x6 = torch.cat([p6, x4], dim=1)
        x6 = self.conv6(x6)
        p7 = self.up_pool7(x6)
        x7 = torch.cat([p7, x3], dim=1)
        x7 = self.conv7(x7)
        p8 = self.up_pool8(x7)
        x8 = torch.cat([p8, x2], dim=1)
        x8 = self.conv8(x8)
        p9 = self.up_pool9(x8)
        x9 = torch.cat([p9, x1], dim=1)
        x10 = self.fc(x9)
        output = nn.functional.upsample(x10, (h, w), mode='bilinear', align_corners=True)

        return output
