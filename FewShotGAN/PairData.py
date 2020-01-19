#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:05:31 2019

@author: dragon
"""
import os
import torch as t
import os.path as osp
import torchvision as tv
from PIL import Image
from random import randint
from FewShotGAN.config import config

transforms    = tv.transforms.Compose([
                tv.transforms.Resize(config.image_size),
                tv.transforms.CenterCrop(config.image_size),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

transforms_gray= tv.transforms.Compose([
                tv.transforms.Resize(config.image_size),
                tv.transforms.CenterCrop(config.image_size),
                tv.transforms.ToTensor(),
                ])
#untransforms = tv.transforms.ToPILImage()

def random_crop(img, width, height):
    width1 = randint(0, img.size[0] - width )    #width1 = randint(img.size[0]/2, 2*img.size[0]/3)
    height1 = randint(0, img.size[1] - height)   #height1 = randint(img.size[1]/3, 2*img.size[1]/3)
    width2 = width1 + width
    height2 = height1 + height
    img = img.crop((width1, height1, width2, height2))
    return img
# This function crops a random template image
        
def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    print(new_arr.size)
    return new_arr    
    
class PairData(t.utils.data.Dataset):
    def __init__(self, root, transforms=transforms):
        imgs_U = os.listdir(osp.join(root, 'Ultrasound')) # Full path   
        #imgs_S = os.listdir(root + 'Sketch')  
        self.imgs_U = [osp.join(osp.join(root, 'Ultrasound'), img) for img in imgs_U]   
        self.imgs_S = [os.path.join((root + 'Sketch'), img[:len(img)-5] + 'B' + img[len(img)-4:]) for img in imgs_U] 
        self.transforms=transforms
        self.transforms_gray=transforms_gray
        
    def __getitem__(self, index):
        imgU_path = self.imgs_U[index]
        imgS_path = self.imgs_S[index]
        pil_imgU = Image.open(imgU_path)
        pil_imgS = Image.open(imgS_path).convert('L')
        
        data_Ultrasound = self.transforms(pil_imgU)
        data_Sketch = self.transforms_gray(pil_imgS)
        return data_Ultrasound, data_Sketch
    
    def __len__(self):
        return len(self.imgs_U)




