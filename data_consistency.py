#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import parser_ops
import utils
import matplotlib.pyplot as plt

parser = parser_ops.get_parser()
args = parser.parse_args()

#mask, sens_maps are torch tensors
dev = "cuda:0"

class data_consistency(): 
    def __init__(self, sens_maps,mask):
        self.shape_list = mask.shape
        self.sens_maps = sens_maps    #(row, column, coil)
        self.mask = mask
        self.scalar = torch.tensor(torch.sqrt(torch.tensor(args.ncol_GLOB * args.nrow_GLOB))).to(dev) #torch.tensor([torch.sqrt(torch.tensor([self.shape_list[1] * self.shape_list[2]])) + 0.j]).to(dev)
                       

    def Ehe_op(self, img, mu_im, mu_k):
        coil_imgs = self.sens_maps * torch.unsqueeze(img, -1) # coil_imgs.shape:  
        kspace = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(coil_imgs, dim=(2,3)), dim=(2,3)), dim=(2,3))/self.scalar 
        masked_kspace = kspace * torch.unsqueeze(self.mask, -1)
        image_space_coil_imgs = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(masked_kspace, dim=(2,3)), dim=(2,3)), dim=(2,3)) * self.scalar
        image_space_comb = torch.sum(image_space_coil_imgs * torch.conj(self.sens_maps), -1)
        
        ispace = image_space_comb + mu_im *img + mu_k * img
        return ispace
    
    def SSDU_kspace(self, img):
        
        
        coil_imgs = self.sens_maps * torch.unsqueeze(img, -1)
        kspace = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(coil_imgs, dim=(2,3)), dim=(2,3)), dim=(2,3)) 
        kspace = kspace / self.scalar
        masked_kspace = kspace * torch.unsqueeze(self.mask, -1)
        
        return masked_kspace
    
def conj_grad(rhs, sens_maps, mask, mu_im, mu_k):
     
     """
Parameters
----------
input_data : contains tuple of  reg output rhs = E^h*y + mu*z , sens_maps and mask
rhs = batch x nrow x ncol x 2
sens_maps : coil sensitivity maps ncoil x nrow x ncol
mask : nrow x ncol
mu : penalty parameter
Encoder : Object instance for performing encoding matrix operations
Returns

!!!!!!!!!!!remember to modify to allow arbitrary batch sizes
-------
data consistency output, nrow x ncol x 2
"""  

     rhs = torch.view_as_complex(rhs) # rhs.shape: shot, batch, row, column
     #rhs = torch.reshape(rhs, (-1, rhs.shape[2], rhs.shape[3])) # rhs.shape: shot * batch, row, column
     encoder = data_consistency(sens_maps, mask)
     x = torch.zeros_like(rhs)
     r, p = rhs, rhs
     rsold = torch.sum(torch.conj(r) * r, dim=(2,3)).cfloat() # shape: batch size
     for i in range(args.CG_Iter):
         Ap = encoder.Ehe_op(p, mu_im, mu_k)
         alpha = torch.tensor(rsold / (torch.sum(torch.conj(p) * Ap, dim=(2,3)).cfloat()) + 0.j)         
         alpha = alpha[:, :, None, None]
         x = x + alpha * p
         r = r - alpha * Ap
         rsnew = torch.sum(torch.conj(r) * r, dim=(2,3)).cfloat()
         beta = rsnew / rsold
         rsold = rsnew
         
         beta = torch.tensor(beta + 0.j)
         beta = beta[:, :, None, None]
         p = r + beta * p
     
     #x = torch.reshape(x, (2, -1, args.nrow_GLOB, args.ncol_GLOB)) # x.shape: shot, batch, row, column
     x = torch.view_as_real(x)
     
     return x
    
    
   