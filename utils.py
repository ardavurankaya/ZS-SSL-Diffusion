#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import h5py

def uniform_selection(input_data, input_mask, rho=0.2, small_acs_block=(4, 4)):

    nrow, ncol = input_data.shape[0], input_data.shape[1]
       
    center_kx = int(find_center_ind(input_data, axes=(1, 2)))
    center_ky = int(find_center_ind(input_data, axes=(0, 2)))

    temp_mask = np.copy(input_mask)
    temp_mask[center_kx - small_acs_block[0] // 2: center_kx + small_acs_block[0] // 2,
    center_ky - small_acs_block[1] // 2: center_ky + small_acs_block[1] // 2] = 0

    pr = np.ndarray.flatten(temp_mask)
    ind = np.random.choice(np.arange(nrow * ncol),
                            size=np.int(np.count_nonzero(pr) * rho), replace=False, p=pr / np.sum(pr))

    [ind_x, ind_y] = index_flatten2nd(ind, (nrow, ncol))

    loss_mask = np.zeros_like(input_mask)
    loss_mask[ind_x, ind_y] = 1

    trn_mask = input_mask - loss_mask

    return trn_mask, loss_mask


def getPSNR(ref, recon):
    """
    Measures PSNR between the reference and the reconstructed images
    """

    mse = np.sum(np.square(np.abs(ref - recon))) / ref.size
    psnr = 20 * np.log10(np.abs(ref.max()) / (np.sqrt(mse) + 1e-10))

    return psnr


def fft(ispace, axes=(0, 1), norm=None, unitary_opt=True):
    """
    Parameters
    ----------
    ispace : coil images of size nrow x ncol x ncoil.
    axes :   The default is (0, 1).
    norm :   The default is None.
    unitary_opt : The default is True.
    Returns
    -------
    transform image space to k-space.
    """

    kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(ispace, axes=axes), axes=axes, norm=norm), axes=axes)

    if unitary_opt:

        fact = 1

        for axis in axes:
            fact = fact * kspace.shape[axis]

        kspace = kspace / np.sqrt(fact)

    return kspace


def ifft(kspace, axes=(0, 1), norm=None, unitary_opt=True):
    """
    Parameters
    ----------
    ispace : image space of size nrow x ncol x ncoil.
    axes :   The default is (0, 1).
    norm :   The default is None.
    unitary_opt : The default is True.
    Returns
    -------
    transform k-space to image space.
    """

    ispace = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(kspace, axes=axes), axes=axes, norm=norm), axes=axes)

    if unitary_opt:

        fact = 1

        for axis in axes:
            fact = fact * ispace.shape[axis]

        ispace = ispace * np.sqrt(fact)

    return ispace


def norm(tensor, axes=(0, 1, 2), keepdims=True):
    """
    Parameters
    ----------
    tensor : It can be in image space or k-space.
    axes :  The default is (0, 1, 2).
    keepdims : The default is True.
    Returns
    -------
    tensor : applies l2-norm .
    """
    for axis in axes:
        tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)

    if not keepdims: return tensor.squeeze()

    return tensor


def find_center_ind(kspace, axes=(1, 2, 3)):
    """
    Parameters
    ----------
    kspace : nrow x ncol x ncoil.
    axes :  The default is (1, 2, 3).
    Returns
    -------
    the center of the k-space
    """

    center_locs = norm(kspace, axes=axes).squeeze()

    return np.argsort(center_locs)[-1:]


def index_flatten2nd(ind, shape):
    """
    Parameters
    ----------
    ind : 1D vector containing chosen locations.
    shape : shape of the matrix/tensor for mapping ind.
    Returns
    -------
    list of >=2D indices containing non-zero locations
    """

    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))

    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]


def sense1(input_kspace, sens_maps, axes=(0, 1)):
    """
    Parameters
    ----------
    input_kspace : nrow x ncol x ncoil
    sens_maps : nrow x ncol x ncoil
    axes : The default is (0,1).
    Returns
    -------
    sense1 image
    """
    
        
    image_space = ifft(input_kspace, axes=axes, norm=None, unitary_opt=True)        
    Eh_op = np.conj(sens_maps) * image_space
    sense1_image = np.sum(Eh_op, axis=axes[-1] + 1)
    
    
    return sense1_image




def complex2real(input_data):
    """
    Parameters
    ----------
    input_data : row x col
    dtype :The default is np.float32.
    Returns
    -------
    output : row x col x 2
    """

    return np.stack((input_data.real, input_data.imag), axis=-1)


def real2complex(input_data):
    """
    Parameters
    ----------
    input_data : row x col x 2
    Returns
    -------
    output : row x col
    """

    return input_data[..., 0] + 1j * input_data[..., 1]

def calculate_rmse(ref_im, im): # Computes the rmse between the reference image and ground truth
                              # return the rmse map and also mean rmse
    ref = np.abs(ref_im) 
    i = np.abs(im) 
    error_map = (ref - i) ** 2
    error = np.sum(error_map) / np.sum(ref**2)
    return np.sqrt(error_map), np.sqrt(error)

def getPSNR(ref_im, im):
    mse = np.sum(np.square(np.abs(ref_im) - np.abs(im))) / ref_im.size
    psnr = 20 * np.log10(np.abs(ref_im.max()) / (np.sqrt(mse) + 1e-16 ))
    return psnr
    

def load_data(data_dir, direction, slice_select, load_sense=False):
    if direction < 10:
        direct = '0' + str(direction)
    else:
        direct = str(direction)
    kdata_dir = data_dir + '/' + 'kdata_' + 'd' + direct + '_' + 'g1.mat'
    
    k = h5py.File(kdata_dir, 'r')
    
    
    kspace = k.get('kdata')
    kspace = np.array(kspace)
    kspace = kspace[0:None:2, :, slice_select, :, :]
    kspace = kspace['real'] + kspace['imag'] * np.array([1.j])
    
    if load_sense:
        sensdata_dir = data_dir + '/csm.mat'
        s = h5py.File(sensdata_dir, 'r')
        sens_map = s.get('sens')
        sens_map = np.array(sens_map)
        sens_map = sens_map[:, slice_select, :, :]
        sens_map = sens_map['real'] + sens_map['imag'] * np.array([1.j])
        return kspace.transpose(0, 3, 2, 1), sens_map.transpose(2, 1, 0)
    
    return kspace.transpose(0, 3, 2, 1)
    
def generate(shot, kspace, mask, sens_maps,rho_val):
    """
    shot: shot number
    mask: data matrix that contains sampling patterns for all shots (shot x row x column)
    kspace: (row x column x coil)
    """
    trn_mask, loss_mask = uniform_selection(kspace, mask[shot], rho=rho_val) # trn_mask: (row x column)
    sub_kspace = kspace * np.expand_dims(trn_mask, 2)
    nw_input = sense1(sub_kspace, sens_maps)
    return nw_input, trn_mask, loss_mask
    
    

