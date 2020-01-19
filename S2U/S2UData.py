#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 20:15:26 2019

@author: dragon
"""
import os
import torch as t
import numpy as np
import torchvision as tv
from PIL import Image
from random import randint
from FewShotGAN.config import config


transforms = tv.transforms.Compose([
                    tv.transforms.Scale(config.image_size),
                    tv.transforms.CenterCrop(config.image_size),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
    
dataset = tv.datasets.ImageFolder(config.S2U_path,transform=transforms)

S2Uloader = t.utils.data.DataLoader(dataset,
                                    batch_size = config.batch_size,
                                    shuffle = True,
                                    num_workers= config.num_workers,
                                    drop_last=True
                                    )
    
