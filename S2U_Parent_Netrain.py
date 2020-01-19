#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 18:22:38 2019

@author: dragon
"""
import os
import pdb                      # break point code: pdb.set_trace()
import tqdm
import time
import torch as t
import torchvision as tv
#from unet import UNet
from FewShotGAN.model import NetD,NetG
from S2U.S2U_model import Generator,Discriminator
from S2U.S2UData import S2Uloader
from FewShotGAN.config import config, class_balanced_cross_entropy_loss, adjust_learning_rate
from S2U.S2U_loss import GeneratorLoss
from FewShotGAN.visualize import Visualizer
from torch.autograd import Variable
from torchnet.meter import AverageValueMeter   

if __name__ == '__main__':
    if config.vis:
        vis = Visualizer(config.env_S2U)  
    
    netg = NetG(None)  # netg = UNet(n_channels=3, n_classes=3) Generator(1) '--upscale_factor', default=4, type=int, choices=[2, 4, 8]
    netd = NetD(None) 
    #generator_criterion = GeneratorLoss()
    
    map_location=lambda storage, loc: storage
    if config.netg_path:
        netg.load_state_dict(t.load(config.netg_path, map_location = map_location)) 
        print('Successed in loading the model G...')
    if config.netd_path:
        netd.load_state_dict(t.load(config.netd_path, map_location = map_location))  
        print('Successed in loading the model D...')
        
    # Optimizer definition
    optimizer_g = t.optim.Adam(list(netg.parameters()), config.lr_g, betas=(config.beta1, 0.999))
    optimizer_d = t.optim.Adam(list(netg.parameters()), config.lr_d, betas=(config.beta1, 0.999))
    criterion = t.nn.BCELoss().cuda()
    # Loss Plot
    errorg_meter = AverageValueMeter()
    errord_meter = AverageValueMeter()
    noises = t.randn(config.batch_size,config.nz,1,1)  #Variable(t.randn(config.batch_size, config.nz, 1, 1))
    true_labels = Variable(t.ones(config.batch_size))
    fake_labels = Variable(t.zeros(config.batch_size,))
    if config.gpu:
        netg.cuda()
        netd.cuda()
        #generator_criterion.cuda()
        noises   = noises.cuda()
        true_labels = true_labels.cuda()
        fake_labels = fake_labels.cuda()
    
    epochs = range(config.max_epoch)
    for epoch in iter(epochs):
        for ii,(img,_) in tqdm.tqdm(enumerate(S2Uloader)):
            img = Variable(img)
            if config.gpu: 
                real_img = img.cuda()
            
            # Training the Discriminator
            if ii % config.d_every==0:
                #Startd_time = time.time()
                optimizer_d.zero_grad()
                #output = netd(real_img)
                #error_d_real = criterion(output,true_labels)
                output = netd(real_img)
                error_d_real = criterion(output,true_labels)
                error_d_real.backward()
                
                noises.data.copy_(t.randn(config.batch_size, config.nz, 1, 1))
                fake_img = netg(noises).detach() 
                output = netd(fake_img)
                error_d_fake = criterion(output,fake_labels)
                error_d_fake.backward()
                optimizer_d.step()
                #output = netd(fake_img)
                #error_d_fake = criterion(output,fake_labels)
                #error_d = error_d_fake + error_d_real
                #error_d.backward()
                error_d = error_d_fake + error_d_real
                errord_meter.add(error_d.data.cpu().numpy())
                #Endd_time = time.time()
                #print('Discriminator time is:')
                #print(Endd_time - Startd_time)
                
                
            # Training the Generator
            if ii % config.g_every==0:
                #Startg_time = time.time()
                optimizer_g.zero_grad()
                noises.data.copy_(t.randn(config.batch_size, config.nz, 1, 1))
                fake_img = netg(noises)
                output = netd(fake_img)
                error_g = criterion(output,true_labels)
                error_g.backward()
                optimizer_g.step()                
                errorg_meter.add(error_g.data.cpu().numpy())
                #Endg_time = time.time()
                #print('Generator time is:')
                #print(Endg_time - Startg_time)
                #error_g = generator_criterion(fake_out, fake_img, real_img)
                #error_g.backward()
                #optimizer_g.step()
                #errorg_meter.add(error_g.data.cpu().numpy())
        
                
            # Visualize the important information
            if config.vis and ii%config.plot_every == 1:
                # Visualize the fake image(Generator)
                vis.images(fake_img.data.cpu().numpy()[:64]*0.5+0.5, win='Generated_Image')
                vis.images(real_img.data.cpu().numpy()[:64]*0.5+0.5, win='Real_Image')
                vis.plot('errorg',errorg_meter.value()[0])
                vis.plot('errord',errord_meter.value()[0])
                
        if epoch % config.decay_every==0:
            # save model and picture
            if not os.path.exists('checkpoints/S2U'):
                os.makedirs('checkpoints/S2U')
            tv.utils.save_image(fake_img.data[:64],'%s/%s.png' %(config.S2U_save_path,epoch),normalize=True,range=(-1,1))
            t.save(netd.state_dict(),'checkpoints/S2U/netd_%s.pth' %epoch)
            t.save(netg.state_dict(),'checkpoints/S2U/netg_%s.pth' %epoch)
            errord_meter.reset()
            errorg_meter.reset()
            optimizer_g = t.optim.Adam(netg.parameters(),config.lr_g,betas=(config.beta1, 0.999))
            optimizer_d = t.optim.Adam(netd.parameters(),config.lr_d,betas=(config.beta1, 0.999))
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                