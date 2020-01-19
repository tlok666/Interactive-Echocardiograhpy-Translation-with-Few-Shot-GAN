#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 10:41:20 2019

@author: dragon
"""
import os
import pdb                      # break point code: pdb.set_trace()
import tqdm
import time
import numpy
import torch
import torch as t
import scipy.io as sio
import torchvision as tv
from unet import UNet
from FewShotGAN.model import NetD,NetG,NetG_Encoding,Encoder_Decoder
from S2U.S2U_model import Generator,Discriminator
from FewShotGAN.PairData import PairData
from FewShotGAN.config import config, class_balanced_cross_entropy_loss, adjust_learning_rate
from S2U.S2U_loss import GeneratorLoss
from FewShotGAN.visualize import Visualizer
from torch.autograd import Variable
from torchnet.meter import AverageValueMeter   
from FewShotGAN.Index import SSIM, compute_iou

Max_Test_Epoch = 300
if __name__ == '__main__':
    if config.vis:
        vis = Visualizer(config.env_FST)  
    
    datapair = PairData(config.Few_Path)
    dataloader = t.utils.data.DataLoader(datapair,
                                         batch_size = 1,#config.batch_size
                                         shuffle = True,
                                         num_workers= 1,
                                         drop_last=True
                                         )   
    
    Testpair = PairData(config.Test_Path)
    Testloader = t.utils.data.DataLoader(Testpair,
                                         batch_size = 1,#config.batch_size
                                         shuffle = True,
                                         num_workers= 1,
                                         drop_last=True
                                         )   
    SSIM_index = SSIM()
    Save_SSIM_Curve = numpy.zeros(Max_Test_Epoch)
    Save_IOU_Curve = numpy.zeros(Max_Test_Epoch)
    
    if config.gpu:
        #netG_U2S = Encoder_Decoder(isFinetuning = True).cuda()
        netG_U2S = torch.nn.DataParallel(Encoder_Decoder()).cuda()
        netD_U2S = NetD(None, 4).cuda()
        #=---------
        netG_S2U_Encoding = NetG_Encoding(1).cuda()
        netG_S2U = NetG(None).cuda()
        netD_S2U = NetD(None, 4).cuda()
        criterionGAN = t.nn.MSELoss().cuda()
        criterionL1 = t.nn.L1Loss().cuda()
        criterion_perceptual_TVL1 = GeneratorLoss().cuda()
    else:
        netG_U2S = torch.nn.DataParallel(Encoder_Decoder())
        netD_U2S = NetD(4)
        #=---------
        netG_S2U_Encoding=NetG_Encoding(1)
        netG_S2U = NetG(IsFineTuning = True)
        netD_S2U = NetD(4).cuda()
        criterionGAN = t.nn.MSELoss()
        criterionL1 = t.nn.L1Loss()
        criterion_perceptual_TVL1 = GeneratorLoss()
    
    map_location=lambda storage, loc: storage
    if config.netG_S2U_pre_path:
        netG_S2U_Encoding.load_state_dict(t.load(config.netG_S2U_pre_path, map_location = map_location)) 
        print('Successed in loading the model S2U_pre Parent Network...')
    if config.netg_path:
        netG_S2U.load_state_dict(t.load(config.netg_path, map_location = map_location)) 
        print('Successed in loading the model S2U Generater Parent Network...')
    if config.netdS2U_path:
        netD_S2U.load_state_dict(t.load(config.netdS2U_path, map_location = map_location)) 
        print('Successed in loading the model S2U Discriminator Parent Network...')
    if config.netU2S_path:
        netG_U2S.load_state_dict(t.load(config.netU2S_path, map_location = map_location)) 
        print('Successed in loading the model U2S Parent Network...')
        
        
    # Optimizer definition
    optimizerS2U_g = t.optim.Adam(list(netG_S2U.parameters()), config.lr_g, betas=(config.beta1, 0.9))
    optimizerS2U_d = t.optim.Adam(list(netD_S2U.parameters()), config.lr_d*5, betas=(config.beta1, 0.9))
    optimizerU2S_g = t.optim.Adam(list(netG_U2S.parameters()), config.lr_g, betas=(config.beta1, 0.9))
    optimizerU2S_d = t.optim.Adam(list(netD_U2S.parameters()), config.lr_d*5, betas=(config.beta1, 0.9))
    # Loss Plot
    errorU2S_gmeter = AverageValueMeter()
    errorU2S_dmeter = AverageValueMeter()
    errorS2U_gmeter = AverageValueMeter()
    errorS2U_dmeter = AverageValueMeter()
    errorSSIM_dmeter = AverageValueMeter()
    errorIOU_dmeter = AverageValueMeter()
    true_labels = Variable(t.ones(config.batch_size))
    fake_labels = Variable(t.zeros(config.batch_size,))
    if config.gpu:
        true_labels = true_labels.cuda()
        fake_labels = fake_labels.cuda()
    
    
    epochs = range(Max_Test_Epoch)
    for epoch in iter(epochs):
        for ii,(img_Ultrasound, img_Sketch) in tqdm.tqdm(enumerate(dataloader)):
            img_Ultrasound = Variable(img_Ultrasound)
            img_Sketch = Variable(img_Sketch)
            if config.gpu: 
                img_Ultrasound = img_Ultrasound.cuda()
                img_Sketch = img_Sketch.cuda()
            
            ###########-----------------------------###########
            ###########-----Ultrasound 2 Sketch-----###########
            ###########-----------------------------###########
            fake_S = netG_U2S(img_Ultrasound)  # G(A)
            # Training the U2S_Discriminator
            if ii%config.d_every==0:
                optimizerU2S_d.zero_grad()
                #----------------U2S
                fake_U2S = t.cat((img_Ultrasound, fake_S), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
                pred_fake = netD_U2S(fake_U2S)
                loss_D_fake = criterionGAN(pred_fake, fake_labels)
                # Real
                real_AB = t.cat((img_Ultrasound, img_Sketch), 1)
                pred_real = netD_U2S(real_AB)
                loss_D_real = criterionGAN(pred_real, true_labels)
                # combine loss and calculate gradients
                loss_D = (loss_D_fake + loss_D_real) * 0.5
                #----------------U2S
                loss_D.backward()
                optimizerU2S_d.step()
                errorU2S_dmeter.add(loss_D.data.cpu().numpy())
            fake_S = netG_U2S(img_Ultrasound)  # G(A)
            # Training the U2S_Generator
            if ii % config.g_every==0:
                optimizerU2S_g.zero_grad()
                #----------------U2S
                fake_U2S = t.cat((img_Ultrasound, fake_S), 1)
                pred_fake = netD_U2S(fake_U2S)
                loss_G_GAN = criterionGAN(pred_fake, true_labels)
                # Second, G(A) = B
                loss_G_L1 = criterionL1(fake_S, img_Sketch) * 100
                # combine loss and calculate gradients
                loss_G = loss_G_GAN + loss_G_L1
                #----------------U2S        
                loss_G.backward()
                optimizerU2S_g.step()                
                errorU2S_gmeter.add(loss_G.data.cpu().numpy())
            ###########-----------------------------###########
            ###########-----Sketch 2 Ultrasound-----###########
            ###########-----------------------------###########
            Sketch_coder = netG_S2U_Encoding(img_Sketch)
            fake_U = netG_S2U(Sketch_coder)  # G(A)
            # Training the S2U_Discriminator
            if ii%config.d_every==0:
                optimizerS2U_d.zero_grad()
                #----------------S2U
                # Fake
                fake_U2S = t.cat((img_Sketch, fake_U), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
                pred_fake = netD_S2U(fake_U2S)
                loss_D_fake = criterionGAN(pred_fake, fake_labels)
                # Real
                real_AB = t.cat((img_Sketch, fake_U), 1)
                pred_real = netD_S2U(real_AB)
                loss_D_real = criterionGAN(pred_real, true_labels)
                # combine loss and calculate gradients
                loss_D = (loss_D_fake + loss_D_real) * 0.5
                #----------------S2U
                loss_D.backward()
                optimizerS2U_d.step()
                errorS2U_dmeter.add(loss_D.data.cpu().numpy())
            Sketch_coder = netG_S2U_Encoding(img_Sketch)
            fake_U = netG_S2U(Sketch_coder)  # G(A)
            # Training the S2U_Generator
            if ii % config.g_every==0:
                optimizerS2U_g.zero_grad()
                #----------------S2U
                fake_U2S = t.cat((img_Sketch, fake_U), 1)
                pred_fake = netD_S2U(fake_U2S)
                loss_G_GAN = criterionGAN(pred_fake, true_labels)
                # Second, G(B) = A
                loss_G_L1 = criterionL1(fake_U, img_Ultrasound) * 100
                ## combine loss and calculate gradients
                loss_G = loss_G_GAN + loss_G_L1
                loss_G += criterion_perceptual_TVL1(pred_fake, fake_U, img_Ultrasound)
                #----------------S2U       
                loss_G.backward()
                optimizerS2U_g.step()                
                errorS2U_gmeter.add(loss_G.data.cpu().numpy())
                
                
                
        # Visualize the important information
        if config.vis and epoch%config.plot_every == 0:
            # Visualize the fake image(U2S Generator)
            vis.images(fake_S.data.cpu().numpy()[:64]*255, win='Fake_U2S')
            vis.images(img_Sketch.data.cpu().numpy()[:64]*255, win='Real_U2S')
            vis.plot('U2S_g',errorU2S_gmeter.value()[0])
            vis.plot('U2S_d',errorU2S_dmeter.value()[0])
            # Visualize the fake image(S2U Generator)
            vis.images(fake_U.data.cpu().numpy()[:64]*127+127, win='Fake_S2U')
            vis.images(img_Ultrasound.data.cpu().numpy()[:64]*127+127, win='Real_S2U')
            vis.plot('S2U_g',errorS2U_gmeter.value()[0])
            vis.plot('S2U_d',errorS2U_dmeter.value()[0])
            
            #--Test
            index_ssim = []
            index_ssim_sum = 0
            index_IOU = []
            index_IOU_sum = 0
            for ii,(img_Ultrasound, img_Sketch) in tqdm.tqdm(enumerate(Testloader)):
                img_Ultrasound = Variable(img_Ultrasound)
                img_Sketch = Variable(img_Sketch)
                if config.gpu: 
                    img_Ultrasound = img_Ultrasound.cuda()
                    img_Sketch = img_Sketch.cuda()
                fake_S = netG_U2S(img_Ultrasound)
                fake_U_coder = netG_S2U_Encoding(img_Sketch)
                fake_U = netG_S2U(fake_U_coder)
                index_ssim.append(SSIM_index(fake_U, img_Ultrasound))
                index_ssim_sum += SSIM_index(fake_U, img_Ultrasound)
                index_IOU.append(compute_iou(fake_S, img_Sketch))
                index_IOU_sum += compute_iou(fake_S, img_Sketch)
            errorSSIM_dmeter.add((index_ssim_sum/len(index_ssim)).detach().cpu().numpy())
            errorIOU_dmeter.add(compute_iou(fake_S, img_Sketch).detach().cpu().numpy())
            
            vis.image((fake_S <= 0.1).float(), win='Compare1')
            vis.image((img_Sketch != 1).float(), win='Compare2')

            vis.plot('SSIM', errorSSIM_dmeter.value()[0])
            vis.plot('IOU', errorIOU_dmeter.value()[0])
            Save_SSIM_Curve[epoch] = errorSSIM_dmeter.value()[0]
            Save_IOU_Curve[epoch] = errorIOU_dmeter.value()[0]
            
        if epoch % config.decay_every==0:
            # save model and picture
            if not os.path.exists(config.U2S_save_path):
                os.makedirs(config.U2S_save_path)
            if not os.path.exists(config.S2U_save_path):
                os.makedirs(config.S2U_save_path)
            if not os.path.exists('checkpoints/FewShot'):
                os.makedirs('checkpoints/FewShot')
            ###########-----Ultrasound 2 Sketch-----###########
            tv.utils.save_image(fake_S.data[:64],'%s/%s.png' %(config.U2S_save_path,epoch),normalize=True,range=(-1,1))
            t.save(netG_U2S.state_dict(),'checkpoints/FewShot/netg_U2S_%s.pth' %epoch)
            t.save(netD_U2S.state_dict(),'checkpoints/FewShot/netd_U2S_%s.pth' %epoch)
            errorU2S_dmeter.reset()
            errorU2S_gmeter.reset()
            optimizerU2S_g = t.optim.Adam(netG_U2S.parameters(),config.lr_g,betas=(config.beta1, 0.999))
            optimizerU2S_d = t.optim.Adam(netD_U2S.parameters(),config.lr_d,betas=(config.beta1, 0.999))
            ###########-----Sketch 2 Ultrasound-----###########
            tv.utils.save_image(fake_U.data[:64],'%s/%s.png' %(config.S2U_save_path,epoch),normalize=True,range=(-1,1))
            t.save(netG_S2U.state_dict(),'checkpoints/FewShot/netg_S2U_%s.pth' %epoch)
            t.save(netD_S2U.state_dict(),'checkpoints/FewShot/netd_S2U_%s.pth' %epoch)
            errorS2U_dmeter.reset()
            errorS2U_gmeter.reset()
            optimizerS2U_g = t.optim.Adam(netG_S2U.parameters(),config.lr_g,betas=(config.beta1, 0.999))
            optimizerS2U_d = t.optim.Adam(netD_S2U.parameters(),config.lr_d,betas=(config.beta1, 0.999))
    sio.savemat('result/Curve_SSIM1.mat', {'Curve': Save_SSIM_Curve})
    sio.savemat('result/Curve_IOU1.mat', {'Curve': Save_IOU_Curve})
    
    
    
    
    
    
    
    
    
    
    
    