# Utilities
# Leon Cai
# MASI Lab
# July 21, 2022

# Set Up

import numpy as np
import contextlib

# Default Context (as opposed to torch.no_grad())

@contextlib.contextmanager
def default_context():
    yield

# Function Definitions

def vox2step(streamline_vox):

    streamline_step = np.diff(streamline_vox, axis=0)
    streamline_step = streamline_step / np.sqrt(np.sum(streamline_step ** 2, axis=1, keepdims=True))
    return streamline_step

def vox2trid(vox):

    trid = vox - np.floor(vox) # this can be parallelized (aka already is)...
    return trid

def vox2trii(vox, img): # ...can we parallelize this? We can pack the sequence and then run through this and then unpack it!

    offset = np.transpose(np.array([[[0, 0, 0], 
                                     [1, 0, 0], 
                                     [0, 1, 0],
                                     [0, 0, 1], 
                                     [1, 1, 0],  
                                     [1, 0, 1], 
                                     [0, 1, 1],
                                     [1, 1, 1]]]), axes=(0, 2, 1)).astype(int)
    tric = np.expand_dims(np.floor(vox).astype(int), axis=2) + offset
    trii = np.stack([coor2idx(tric[:, :, c], img) for c in range(8)], axis=1)
    return trii

def coor2idx(coor, img): # Returns index of -1 for invalid coordinates

    invalid_idx = np.logical_or(np.any(coor < 0, axis=1), np.any(coor > img.shape, axis=1))
    idx = np.ravel_multi_index(tuple(np.transpose(coor, axes=(1, 0))), img.shape, mode='clip')
    idx[invalid_idx] = -1
    return idx

def triinterp(img, trid, trii, fourth_dim=True):

    assert len(img.shape) == 3 or len(img.shape) == 4, 'img must be a 3D or 4D array (x, y, z, c).'
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=3)
    else:
        fourth_dim = True
    img = np.reshape(img, (-1, img.shape[-1]))

    # Source: https://www.wikiwand.com/en/Trilinear_interpolation

    xd = np.expand_dims(trid[:, 0], axis=1)
    yd = np.expand_dims(trid[:, 1], axis=1)
    zd = np.expand_dims(trid[:, 2], axis=1)
    
    c000 = img[trii[:, 0], :]
    c100 = img[trii[:, 1], :]
    c010 = img[trii[:, 2], :]
    c001 = img[trii[:, 3], :]
    c110 = img[trii[:, 4], :]
    c101 = img[trii[:, 5], :]
    c011 = img[trii[:, 6], :]
    c111 = img[trii[:, 7], :]

    c00 = c000*(1-xd) + c100*xd
    c01 = c001*(1-xd) + c101*xd
    c10 = c010*(1-xd) + c110*xd
    c11 = c011*(1-xd) + c111*xd

    c0 = c00*(1-yd) + c10*yd
    c1 = c01*(1-yd) + c11*yd

    c = c0*(1-zd) + c1*zd

    if not fourth_dim:
        c = c[:, 0]

    return c

def streamline2network(streamline_vox, img):

    streamline_step = vox2step(streamline_vox)          # Cartesian steps 1,...,n-1
    streamline_trid = vox2trid(streamline_vox)          # Distance between two nearest voxels 1,...,n
    streamline_trii = vox2trii(streamline_vox, img)     # Raveled indices for neighboring 8 voxels 1,...,n

    return streamline_trid[:-1, :], streamline_trii[:-1, :], streamline_step

def len2mask(streamlines_length):

    batch_size = len(streamlines_length)
    mask = np.ones((np.max(streamlines_length), len(streamlines_length))) # always batch_first = false
    for b in range(batch_size):
        mask[streamlines_length[b]:, b] = 0
    return mask
