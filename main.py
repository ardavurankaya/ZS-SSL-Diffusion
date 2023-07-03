#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:53:19 2023

@author: ismailardavurankaya
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np
import scipy.io as sio
import os
import utils
import parser_ops
import h5py
import unrollednet
import matplotlib.pyplot as plt
import time


if torch.cuda.is_available():
    dev = "cuda:0"
else:
   dev = "cpu"
   




parser = parser_ops.get_parser()
args = parser.parse_args()

args.num_reps = 10
args.batchSize = 1



slice_select = 15

#save_dir ='/autofs/space/marduk_001/users/Arda/my-ssl/saved_models/multi_shot_25rep'  + '.pth' 
save_dir = '/content/drive/MyDrive/models/multishot/multi_shot_10rep_d22.pth'


print('Loading data  for training............... ')




main_dir = '/content/drive/MyDrive/buda_ms'
args.kdata_dir = main_dir + '/kdata.npy'
args.sensdata_dir = '/content/drive/MyDrive/buda/csm.mat'

kspace_train = np.load(args.kdata_dir)
kspace_train = kspace_train[:, 22]

print(kspace_train.shape)

s = h5py.File(args.sensdata_dir,'r')
sens_maps = s.get('sens')
sens_maps = np.array(sens_maps)

print(sens_maps.shape)

sens_maps = sens_maps[:,slice_select,:,:]
sens_maps = sens_maps['real'] + sens_maps['imag'] * np.array([1.j])
sens_maps = sens_maps.transpose(2,1,0)




args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB  = kspace_train.shape[1:]



original_mask = np.zeros((2, args.nrow_GLOB, args.ncol_GLOB))
original_mask[0, :,56:None:5] = 1
original_mask[1, :, 59:None:5] = 1






print('Normalize the kspace to 0-1 region')

normalization_factor = 9e-5  #np.max(np.abs(kspace_train))

kspace_train= kspace_train / normalization_factor

#%%

#..................Generate validation mask.....................................

cv_trn_mask, cv_val_mask = np.zeros_like(original_mask), np.zeros_like(original_mask)

cv_trn_mask[0], cv_val_mask[0] = utils.uniform_selection(kspace_train[0],original_mask[0], rho=args.rho_val)
cv_trn_mask[1], cv_val_mask[1] = utils.uniform_selection(kspace_train[1],original_mask[1], rho=args.rho_val)


remainder_mask, cv_val_mask=np.copy(cv_trn_mask),np.copy(np.complex64(cv_val_mask))

print('size of kspace: ', kspace_train[np.newaxis,...].shape, ', maps: ', sens_maps.shape, ', mask: ', original_mask.shape)

trn_mask, loss_mask = np.empty((2, args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64), \
                                np.empty((2, args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
# train data
nw_input = np.empty((2, args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
ref_kspace = np.empty((2, args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)
#...............................................................................
# validation data
ref_kspace_val = np.empty((2, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)
nw_input_val = np.empty((2, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

#%%

print('create training&loss masks and generate network inputs... ')
#train data


for i in range(2):
    for j in range(args.num_reps):
        trn_mask[i, j], loss_mask[i, j] = utils.uniform_selection(kspace_train[i],remainder_mask[i], rho=args.rho_train)
        sub_kspace = kspace_train[i] * np.expand_dims(trn_mask[i, j], -1)
        ref_kspace[i, j]= kspace_train[i] * np.expand_dims(loss_mask[i, j], -1)
        nw_input[i, j] = utils.sense1(sub_kspace,sens_maps)    





#..............................validation data.....................................
nw_input_val = utils.sense1(kspace_train * np.expand_dims(cv_trn_mask, -1), sens_maps, axes=(1,2))


ref_kspace_val=kspace_train * np.expand_dims(cv_val_mask, -1)




sens_maps = torch.from_numpy(sens_maps).to(dev)



nw_input = utils.complex2real(nw_input)


nw_input_val = utils.complex2real(nw_input_val)


print('size of ref kspace: ', ref_kspace.shape, ', nw_input: ', nw_input.shape, ', maps: ', sens_maps.shape, ', mask: ', trn_mask.shape)

# %% set the batch size
total_batch = int(np.floor(np.float32(args.num_reps) / (args.batchSize)))


trn_mask = torch.from_numpy(trn_mask)
loss_mask = torch.from_numpy(loss_mask)

cv_val_mask = torch.from_numpy(cv_val_mask).to(dev)
cv_val_mask = torch.unsqueeze(cv_val_mask, 1)

cv_trn_mask = torch.from_numpy(cv_trn_mask).to(dev)
cv_trn_mask = torch.unsqueeze(cv_trn_mask, 1)


trn_loss = 0
model = unrollednet.UnrolledNet(sens_maps, trn_mask, loss_mask).to(dev)


total_loss = []
total_val_loss = []
scalar = torch.tensor([0.5], dtype=torch.float32).to(dev)

args.learning_rate = 3e-4
optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
lowest_val_loss = np.inf
nw_input = torch.from_numpy(nw_input)



ref_kspace = torch.from_numpy(ref_kspace)
ref_kspace_val = torch.from_numpy(ref_kspace_val).to(dev)
ref_kspace_val = torch.unsqueeze(ref_kspace_val, 1)

nw_input_val = torch.tensor(nw_input_val, dtype=torch.float32)
nw_input_val = nw_input_val.to(dev)
nw_input_val = torch.unsqueeze(nw_input_val, 1)
#%%

output_imgs = torch.empty((2, args.nrow_GLOB, args.ncol_GLOB, 2)).to(dev)


print("Number of trainable parameters is: ****************************")
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

#%%

epoch, val_loss_tracker = 0, 0
weight = torch.tensor([1e-6], dtype=torch.float32).to(dev)
lowest_val_loss = np.inf

t0 = time.time()

while epoch < -10 and val_loss_tracker < 15:
    epoch_loss = 0
    for j in range(total_batch):
        model.set_trn_mask(trn_mask[:, j * args.batchSize : (j+1) * args.batchSize]. to(dev))
        model.set_loss_mask(loss_mask[:, j * args.batchSize : (j+1) * args.batchSize].to(dev))
        output_imgs, nw_output_kspace = model.forward(nw_input[:, j * args.batchSize : (j+1) * args.batchSize].to(dev))[0:2]
        current_kspace = ref_kspace[:, j * args.batchSize : (j+1) * args.batchSize].to(dev)
        trn_loss = torch.mul(scalar, torch.linalg.norm(nw_output_kspace - current_kspace)) / torch.linalg.norm(current_kspace) + torch.mul(1 - scalar, torch.linalg.norm(torch.flatten(current_kspace-nw_output_kspace), ord=1)) / torch.linalg.norm(torch.flatten(current_kspace), ord=1)
        trn_loss += weight * torch.sum(torch.abs(torch.abs(torch.view_as_complex(output_imgs[0])) - torch.abs(torch.view_as_complex(output_imgs[1]))))
        epoch_loss += trn_loss
        optimizer.zero_grad()
        trn_loss.backward()
        optimizer.step()
    
    
    print(f"Training loss in epoch {epoch} is {epoch_loss.item() / total_batch}")
        
    with torch.no_grad():
        model.set_trn_mask(cv_trn_mask)
        model.set_loss_mask(cv_val_mask)
        val_output_kspace = model.forward(nw_input_val)[1]
        val_loss = torch.mul(scalar, torch.linalg.norm(val_output_kspace - ref_kspace_val)) / torch.linalg.norm(ref_kspace_val) + torch.mul(1-scalar, torch.linalg.norm(torch.flatten(ref_kspace_val - val_output_kspace), ord=1)) / torch.linalg.norm(torch.flatten(ref_kspace_val), ord=1)
        print(f"Validation loss in epoch {epoch} is {val_loss.item()}")
        if val_loss < lowest_val_loss:
            print('This is a new minimum of validation loss.')
            lowest_val_loss = val_loss
            val_loss_tracker = 0
            torch.save(model.state_dict(), save_dir)
        else:
            val_loss_tracker += 1
    epoch += 1
    
print('training finished!')
    

t1 = time.time()

print(f'Total training time is {t1-t0}')




#%%

reference_imgs = np.zeros((2, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

print('Load data for testing...')

ref_img_dir = '/content/drive/MyDrive/loraks/loraks_d25_g1.mat'
r = h5py.File(ref_img_dir, 'r')
ref_img = r.get('img')
ref_img = np.array(ref_img)
ref_img = ref_img[[0, 6], slice_select, :, :]
ref_img = ref_img['real'] + ref_img['imag'] * np.array([1.j])
ref_img = ref_img.transpose(0, 2, 1)


#%%

sens_maps = sens_maps.detach().cpu().numpy()
sens_mask = sens_maps > 0
sens_mask = np.sum(sens_mask, axis=-1) > 0             


#%%

nw_input = utils.sense1(kspace_train[:, :, :, :], sens_maps, axes=(1,2))
nw_input = np.expand_dims(nw_input, 1)

model = unrollednet.UnrolledNet(torch.from_numpy(sens_maps).to(dev), torch.from_numpy(original_mask).to(dev), torch.from_numpy(original_mask).to(dev))
model.load_state_dict(torch.load(save_dir))
model = model.to(dev)

print(model.mu_im)
print(model.mu_k)

original_mask = np.expand_dims(original_mask, 1)

model.set_trn_mask(torch.from_numpy(original_mask).to(dev))
model.set_loss_mask(torch.from_numpy(original_mask).to(dev))

network_recon, output_kspace, cg_sense = model.forward(torch.from_numpy(utils.complex2real(nw_input)).to(dev))

network_recon = torch.squeeze(network_recon)
network_recon = network_recon.detach().cpu().numpy()
network_recon = utils.real2complex(network_recon)
network_recon *= normalization_factor
network_recon *= sens_mask

cg_sense = torch.squeeze(cg_sense)
cg_sense = cg_sense.detach().cpu().numpy()
cg_sense = utils.real2complex(cg_sense)
cg_sense *= normalization_factor
cg_sense *= sens_mask

plt.figure()
plt.imshow(np.rot90(np.abs(network_recon[0])), cmap='gray')
plt.savefig('/content/drive/MyDrive/figures/multishot/shot1_10rep_d22.jpg')

plt.figure()
plt.imshow(np.rot90(np.abs(network_recon[1])), cmap='gray')
plt.savefig('/content/drive/MyDrive/figures/multishot/shot2_10rep_d22.jpg')


network_rmse_1 = utils.calculate_rmse(ref_img[0], network_recon[0])[1]
network_rmse_2 = utils.calculate_rmse(ref_img[1], network_recon[1])[1]

sense_rmse_1 = utils.calculate_rmse(ref_img[0], cg_sense[0])[1]
sense_rmse_2 = utils.calculate_rmse(ref_img[1], cg_sense[1])[1]

print(f'Network rmse at shot 1 is equal to {network_rmse_1}')
print(f'Network rmse at shot 2 is equal to {network_rmse_2}')
print(f'Sense rmse at shot 1 is equal to {sense_rmse_1}')
print(f'Sense rmse at shot 2 is equal to {sense_rmse_2}')

