#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 20:26:59 2019

@author: dragon
"""
from torch import nn
from unet.unet_model import UNet

class Encoder_Decoder(nn.Module):
    '''
    Generator
    '''
    def __init__(self, IsFineTuning = False):
        super(Encoder_Decoder, self).__init__()
        self.Network = UNet(3, 1)
        self.output  = nn.Sigmoid()
    
    def forward(self, input):

        AE = self.Network(input)
        output = self.output(AE)
        
        return output
    
class NetD(nn.Module):
    '''
    Discriminator
    '''
    def __init__(self, opt, inChannel = 3):
        super(NetD, self).__init__()
        ndf = 64
        self.main = nn.Sequential(
            nn.Conv2d(inChannel, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  
        )

    def forward(self, input):
        return self.main(input).view(-1)
    
    
class NetG_Encoding(nn.Module):
    '''
    Generator_Encoding
    '''
    def __init__(self, inChannel = 3):
        super(NetG_Encoding, self).__init__()
        ndf = 64
        self.main = nn.Sequential(
            nn.Conv2d(inChannel, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, 100, 4, 1, 0, bias=False),
            nn.Tanh()  
        )

    def forward(self, input):
        return self.main(input)
    
class NetG(nn.Module):
    '''
    Generator
    '''
    def __init__(self, opt, IsFineTuning = False):
        super(NetG, self).__init__()
        ngf = 64
        self.IsFineTuning = IsFineTuning

        self.main0 = nn.Sequential(
            nn.ConvTranspose2d(100, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            )
        self.main1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()  
        )
        self.Encoder = nn.Sequential(
            nn.Conv2d(3, ngf, 5, 3, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.Conv2d(ngf, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.Conv2d(ngf * 4, ngf * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        )

    
    def forward(self, input):
        if self.IsFineTuning:
            output = self.Encoder(input)
            output = self.main1(output)
        else:
            output = self.main1(self.main0(input))
        return output