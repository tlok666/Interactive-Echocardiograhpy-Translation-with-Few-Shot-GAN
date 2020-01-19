#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 08:36:30 2019

@author: dragon
"""
from __future__ import division
import os
import torch
from torch.autograd import Variable
from torch.utils import data
# from resnet import FCN
from FewShotGAN.model import Encoder_Decoder
# from gcn import FCN
from U2S.datasets import VOCDataSet
from U2S.loss import CrossEntropy2d, CrossEntropyLoss2d
from FewShotGAN.visualize import Visualizer
from U2S.transform import ReLabel, ToLabel, ToSP, Scale
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor
import tqdm
from PIL import Image
import numpy as np
from torchnet.meter import AverageValueMeter
from FewShotGAN.config import config

input_transform = Compose([
    Scale((config.image_size, config.image_size), Image.BILINEAR),
    ToTensor(),
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),

])
target_transform = Compose([
    Scale((config.image_size, config.image_size), Image.NEAREST),
    ToSP(config.image_size),
    ToLabel(),
    ReLabel(255, 21),
])

trainloader = data.DataLoader(VOCDataSet(config.U2S_path, img_transform=input_transform,
                                         label_transform=target_transform),
                              batch_size=16, shuffle=True, pin_memory=True)

if torch.cuda.is_available():
    model = torch.nn.DataParallel(Encoder_Decoder())
    model.cuda()

epoches = 8000
lr = 1e-4
weight_decay = 2e-5
momentum = 0.9
weight = torch.ones(22)
weight[21] = 0
max_iters = 92*epoches

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                            weight_decay=weight_decay)
vis = Visualizer(config.env_U2S)  
error_meter = AverageValueMeter()

if os.path.exists(config.netU2S_path):
    #model.load_state_dict(torch.load(config.netU2S_path))
    print('Successed in loading the model G...')
        
model.train()
for epoch in range(epoches):
    running_loss = 0.0
    for i, (images, labels_group) in tqdm.tqdm(enumerate(trainloader)):
        if torch.cuda.is_available():
            images = [Variable(image.cuda()) for image in images]
            labels_group = [labels for labels in labels_group]
        else:
            images = [Variable(image) for image in images]
            labels_group = [labels for labels in labels_group]

        optimizer.zero_grad()
        losses = []
        img = images[0]
        labels=labels_group[0]
        outputs = model(img)
        
        labels_jj = (labels[0] != 0).float().cuda()
        losses.append(criterion(outputs, labels_jj.unsqueeze(1)))

        #if epoch < 40:
        #    loss_weight = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
        #else:
        #    loss_weight = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]

        loss = 0
        #for w, l in zip(loss_weight, losses):
        #    loss += w*l
        for w in range(len(losses)):
            loss += losses[w]
        loss.backward()
        optimizer.step()
        running_loss += loss.data.cpu().numpy()

        # lr = lr * (1-(92*epoch+i)/max_iters)**0.9
        # for parameters in optimizer.param_groups:
        #     parameters['lr'] = lr

    print("Epoch [%d] Loss: %.4f" % (epoch+1, running_loss/i))
    #ploter.plot("loss", "train", epoch+1, running_loss/i)
    error_meter.add(running_loss/i)
    ss = outputs
    vis.images(img[:,2,:,:].data.cpu().numpy()[:64]*127+127, win='img')
    vis.images(ss[:,0,:,:].data.cpu().numpy()[:64]*255, win='Mask1')
    vis.images(labels_jj.data.cpu().float().numpy()*255, win='Gt')
    vis.plot('errorg',error_meter.value()[0])
    running_loss = 0

    if not os.path.exists('checkpoints/U2S'):
        os.makedirs('checkpoints/U2S')
    if (epoch+1) % 10 == 0:
        lr /= 10
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)
        torch.save(model.state_dict(), "checkpoints/U2S/AutoEncoder-%d.pth" % (epoch+1))


torch.save(model.state_dict(), "checkpoints/U2S/AutoEncoder.pth")
