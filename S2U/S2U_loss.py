#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:11:48 2019

@author: dragon
"""

import torch
import torch as t
from torch import nn
from torchvision.models.vgg import vgg16
from torch.autograd import Variable
from FewShotGAN.config import config

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.L1_loss = t.nn.L1Loss()
        self.tv_loss = TVLoss()
        self.true_label = Variable(t.ones(config.Num_shot)).cuda()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        #adversarial_loss = torch.mean(1 - out_labels)
        adversarial_loss = self.mse_loss(out_labels, self.true_label)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.L1_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        #Image_loss is :
        #tensor(0.5063, device='cuda:0', grad_fn=<MseLossBackward>)
        #Adversarial_loss is :
        #tensor(0.0104, device='cuda:0', grad_fn=<MseLossBackward>)
        #Perception_loss is :
        #tensor(0.8342, device='cuda:0', grad_fn=<MseLossBackward>)
        #Tv_loss is :
        #tensor(0.3708, device='cuda:0', grad_fn=<DivBackward0>)
        return 1e-2*perception_loss + 2e-8 * tv_loss #100*image_loss + adversarial_loss + 1e-2*perception_loss + 2e-8 * tv_loss 
    

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
