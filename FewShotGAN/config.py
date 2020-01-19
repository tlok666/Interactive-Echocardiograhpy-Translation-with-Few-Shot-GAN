#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 12:05:47 2019

@author: dragon(long teng)
"""
import torch as t
import numpy as np

def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        print('Learning Rate is:' + str(param_group['lr'] * decay_rate))
# Source code：https://blog.csdn.net/u012436149/article/details/70666068 
        
def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """

    labels = t.ge(label, 0.5).float()

    num_labels_pos = t.sum(labels)
    num_labels_neg = t.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = t.ge(output, 0).float()
    loss_val = t.mul(output, (labels - output_gt_zero)) - t.log(1 + t.exp(output - 2 * t.mul(output, output_gt_zero)))

    loss_pos = t.sum(-t.mul(labels, loss_val))
    loss_neg = t.sum(-t.mul(1.0 - labels, loss_val))
    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= np.prod(label.size())
    elif batch_average:
        final_loss /= label.size()[0]
    return final_loss

class config(object):
        
    S2U_path= '/media/dragon/software/dataset/Ultrasound1'  # dataset path
    U2S_path= '/media/dragon/software/dataset/VOCdevkit'  # dataset path
    Few_Path= 'dataset/FewShot/1shot/'  # dataset path
    env_FST = 'Few_Shot_Transfer_Learning_1shot'         # visdom env
    Test_Path= 'dataset/FewShot/Test/'  # dataset path
    
    num_workers = 4      # multi-thread number to load data
    netd_path = 'checkpoints/S2U/netd_10000.pth' #pretained model
    netg_path = 'checkpoints/S2U/netg_10000.pth'
    netU2S_path = 'checkpoints/U2S/AutoEncoder-2450.pth'      # 'cpkt/netg_xxx.pth'
    netG_S2U_pre_path = None
    #netS2U_path = 'checkpoints/S2U/netg_2900.pth'      # 'checkpoints/S2U/netg_1800.pth'
    netdS2U_path= None
    max_epoch = 20000     # Max epoch number
    decay_every = 200     # save model every 200 epochs
    
    inC = 3     # the channels of input Template Image and Source Image(We assume that, T and S has the same channel number)
    outN= 64    # the output featuremap number
    nz  = 100   # onehot encoder for encoder-decoder
    
    batch_size = 64
    ins_channel = 3
    image_size  = 96
    Num_shot = 1
    
    lr_g = 2e-4  # Learning rate of generator
    lr_d = 2e-4  # Learning rate of discriminator
    beta1=0.5    # beta1 parameter of Adam optimizer
    gpu=True     # If GPU is adopted
    
    S2U_save_path = 'result/S2U/'  #saving path    
    U2S_save_path = 'result/U2S/'  #saving path   
    FewShot_save_path = 'result/FewShot/'  #saving path   
    vis = True             # If use visdom for visualision
    env_S2U = 'S2U_Parent_Netrain'         # visdom env
    env_U2S = 'U2S_Parent_Netrain'         # visdom env
    d_every = 1         # train new discriminator every 5 batchs
    g_every = 15         # train new generator every 5 epoches, Transfer learning every 15 epoches
    plot_every = 2     # plot with visdom every 20 batchs
    adjust_lr  = 20000
    
