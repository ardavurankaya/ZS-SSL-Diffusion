#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np
import data_consistency
import parser_ops
import matplotlib.pyplot as plt


parser = parser_ops.get_parser()
args = parser.parse_args()
dev = "cuda:0"
#dev = "cpu"



class UnrolledNet(nn.Module):
    """
    Parameters
    ----------
    input_x: batch_size x nrow x ncol x 2
    sens_maps: batch_size x ncoil x nrow x ncol
    trn_mask: batch_size x nrow x ncol, used in data consistency units
    loss_mask: batch_size x nrow x ncol, used to define loss in k-space
    args.nb_unroll_blocks: number of unrolled blocks
    args.nb_res_blocks: number of residual blocks in ResNet
    Returns
    ----------
    x: nw output image
    nw_kspace_output: k-space corresponding nw output at loss mask locations
    x0 : dc output without any regularization.
    all_intermediate_results: all intermediate outputs of regularizer and dc units
    mu: learned penalty parameter
    """
    def __init__(self, sens_maps, trn_mask, loss_mask):
        super(UnrolledNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=3, padding='same')

        self.conv4 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding='same')
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=3, padding='same')

        self.resnet_layers_im = nn.ModuleList()
    
        for k in range(args.nb_res_blocks):
            self.resnet_layers_im.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'))
        
        self.resnet_layers_k = nn.ModuleList()
        for k in range(args.nb_res_blocks):
            self.resnet_layers_k.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'))

        self.activate = nn.ReLU()
        self.sens_maps = sens_maps
        self.mu_im = nn.Parameter(torch.tensor([0.05],requires_grad=True))
        self.mu_k = nn.Parameter(torch.tensor([0.01], requires_grad=True))
        self.trn_mask = trn_mask
        self.loss_mask = loss_mask
        self.scalar = torch.tensor(torch.sqrt(torch.tensor(args.ncol_GLOB * args.nrow_GLOB))).to(dev) 
        
    def set_loss_mask(self, mask):
        self.loss_mask = mask
    
    def set_trn_mask(self, mask):
        self.trn_mask = mask
    
    
    def forward(self, x):
        input_x = x  # x.shape: total_shot, batchsize, row, column, 2(real&imag)
        input_k = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(torch.view_as_complex(input_x), dim=(2,3)), dim=(2,3)), dim=(2,3)) / self.scalar
        input_k = torch.view_as_real(input_k)
        k = input_k
        mu_init = torch.tensor([0], dtype=torch.float32).to(dev)
        x0 = data_consistency.conj_grad(x, self.sens_maps, self.trn_mask, 0, 0)
        for i in range(args.nb_unroll_blocks):
            x = torch.stack((x[0, :, :, :, 0], x[0, :, :, :, 1], x[1, :, :, :, 0], x[1, :, :, :, 1]), dim=-1)
            x = x.permute(0, 3, 1, 2) # x.shape: batch, feature(4), row, column
            x = x.float()
            x = self.conv1(x)
            first_layer = x
            for j in range(args.nb_res_blocks):
                previous_layer = x
                m = self.resnet_layers_im[j]
                x = self.activate(m(x))                
                x = m(x)
                x = torch.mul(x, torch.tensor([0.1],dtype=torch.float32).to(dev))
                x = x + previous_layer
                
            rb_output = self.conv2(x)
            temp_output = rb_output + first_layer
            x = self.conv3(temp_output)
            
            
            k = torch.stack((k[0, :, :, :, 0], k[0, :, :, :, 1], k[1, :, :, :, 0], k[1, :, :, :, 1]), dim=-1)
            k = k.permute(0, 3, 1, 2) # x.shape: batch, feature(4), row, column
            k = k.float()
            
            k = self.conv4(k)
            first_layer = k
            for j in range(args.nb_res_blocks):
                previous_layer = k
                m = self.resnet_layers_k[j]
                k = self.activate(m(k))                
                k = m(k)
                k = torch.mul(k, torch.tensor([0.1],dtype=torch.float32).to(dev))
                k = k + previous_layer
                
            rb_output = self.conv5(k)
            temp_output = rb_output + first_layer
            k = self.conv6(temp_output)

            
            x = x.permute(0, 2, 3, 1) # x.shape: batch, row, column, feature
            k = k.permute(0, 2, 3, 1)
            x = torch.stack((x[:, :, :, 0:2], x[:, :, :, 2:4]), dim=0) # shot, batch, row, column
            k = torch.stack((k[:, :, :, 0:2], k[:, :, :, 2:4]), dim=0) # shot, batch, row, column
        
            k = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(torch.view_as_complex(k), dim=(2,3)), dim=(2,3)), dim=(2,3)) * self.scalar
            k = torch.view_as_real(k)
            rhs = input_x + self.mu_im * x + self.mu_k * k
            x = data_consistency.conj_grad(rhs, self.sens_maps, self.trn_mask,self.mu_im, self.mu_k)
            k = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(torch.view_as_complex(x), dim=(2,3)), dim=(2,3)), dim=(2,3)) / self.scalar
            k = torch.view_as_real(k)

        encoder = data_consistency.data_consistency(self.sens_maps, self.loss_mask)
        
        nw_kspace_output = encoder.SSDU_kspace(torch.view_as_complex(x))

        return x, nw_kspace_output, x0
            
                
        
        