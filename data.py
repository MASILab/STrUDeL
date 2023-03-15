# Dataset Definitions
# Leon Cai
# MASI Lab
# August 31, 2022

# Set Up

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import nibabel as nib

from utils import onehot

class DT1Dataset(Dataset):

    def __init__(self, dt1_dirs, num_batches):

        super(DT1Dataset, self).__init__()
        self.dt1_dirs = dt1_dirs
        self.num_batches = num_batches

    def __getitem__(self, index):

        dt1_dir = self.dt1_dirs[index]

        # Try doing more image stuff in RAM

        in_dir = os.path.dirname(dt1_dir)
        fod_file = os.path.join(in_dir, 'dwmri_fod_mni_2mm_trix.nii.gz')
        t1_file = os.path.join(in_dir, 'T1_N4_mni_1mm.nii.gz')
        mask_file = os.path.join(in_dir, 'T1_seed_mni_1mm.nii.gz')
        act_file = os.path.join(in_dir, 'T1_5tt_mni_1mm.nii.gz')
        tseg_file = os.path.join(in_dir, 'T1_tractseg_mni_2mm.nii.gz')
        slant_file = os.path.join(in_dir, 'T1_slant_mni_2mm.nii.gz')
        # posenc_file = os.path.join(in_dir, 'T1_posenc16_mni_2mm.nii.gz')

        fod_img = nib.load(fod_file).get_fdata()
        t1_img = nib.load(t1_file).get_fdata()[:-1, :-1, :-1]
        mask_img = nib.load(mask_file).get_fdata()[:-1, :-1, :-1].astype(bool)
        act_img = nib.load(act_file).get_fdata()[:-1, :-1, :-1, :-1]
        tseg_img = nib.load(tseg_file).get_fdata()
        slant_img = nib.load(slant_file).get_fdata()
        # posenc_img = nib.load(posenc_file).get_fdata()

        fod = torch.FloatTensor(np.expand_dims(np.transpose(fod_img, axes=(3, 0, 1, 2)), axis=0))

        t1_ten = torch.FloatTensor(np.expand_dims(t1_img / np.median(t1_img[mask_img]), axis=(0, 1)))
        act_ten = torch.FloatTensor(np.expand_dims(np.transpose(act_img, axes=(3, 0, 1, 2)), axis=0))
        tseg_ten = torch.FloatTensor(np.expand_dims(np.transpose(tseg_img, axes=(3, 0, 1, 2)), axis=0))
        slant_ten = torch.FloatTensor(np.expand_dims(np.transpose(slant_img, axes=(3, 0, 1, 2)), axis=0))
        # posenc_ten = torch.FloatTensor(np.expand_dims(np.transpose(posenc_img, axes=(3, 0, 1, 2)), axis=0))
        # ten = torch.cat((t1_ten, act_ten, tseg_ten, slant_ten, posenc_ten), dim=1)
        # ten = torch.cat((t1_ten, act_ten, tseg_ten, slant_ten), dim=1)
        ten_1mm = torch.cat((t1_ten, act_ten), dim=1)
        ten_2mm = torch.cat((tseg_ten, slant_ten), dim=1)

        brain = torch.FloatTensor(np.expand_dims(mask_img, axis=(0, 1)))

        # # (OG) Reading images from disk
        # ten = torch.load(os.path.join(dt1_dir, 'ten.pt'))
        # tseg = torch.load(os.path.join(dt1_dir, 'tseg.pt'))
        # slant = torch.load(os.path.join(dt1_dir, 'slant.pt'))
        # ten = torch.cat((ten, tseg, slant), dim=1)
        # # sf = torch.load(os.path.join(dt1_dir, 'sf.pt'))
        # fod = torch.load(os.path.join(dt1_dir, 'fod.pt'))
        # brain = torch.load(os.path.join(dt1_dir, 'mask.pt'))
        # # fs  = onehot(torch.load(os.path.join(dt1_dir, 'fs.pt')), num_classes=115)

        # Streamlines (must be read from disk)

        b = np.random.randint(0, self.num_batches)
        step = torch.load(os.path.join(dt1_dir, 'step_{:06}.pt'.format(b)))
        trid = torch.load(os.path.join(dt1_dir, 'trid_{:06}.pt'.format(b)))
        trii = torch.load(os.path.join(dt1_dir, 'trii_{:06}.pt'.format(b)))
        mask = torch.load(os.path.join(dt1_dir, 'mask_{:06}.pt'.format(b)))
        # tdi  = torch.load(os.path.join(dt1_dir, 'tdi_{:06}.pt'.format(b)))        # These three
        # tdi_weights_fxn = torch.load(os.path.join(dt1_dir, 'tdi_weights_fxn.pt')) # lines do 
        # mask = torch.Tensor(tdi_weights_fxn(tdi)) * mask                          # TDI weighting
        return (ten_1mm, ten_2mm, fod, brain, step, trid, trii, mask)
        # return (ten, step, trid, trii, mask, tdi)
        # return (torch.cat((ten, fs), dim=1), step, trid, trii, mask)

    def __len__(self):

        return len(self.dt1_dirs)

def unload(ten_1mm, ten_2mm, fod, brain, step, trid, trii, mask):
# def unload(ten, step, trid, trii, mask, tdi):

    ten_1mm = ten_1mm[0]
    ten_2mm = ten_2mm[0]
    fod = fod[0]
    brain = brain[0]
    step = step[0]
    trid = nn.utils.rnn.PackedSequence(trid.data[0], batch_sizes=trid.batch_sizes[0], sorted_indices=trid.sorted_indices[0], unsorted_indices=trid.unsorted_indices[0])
    trii = nn.utils.rnn.PackedSequence(trii.data[0], batch_sizes=trii.batch_sizes[0], sorted_indices=trii.sorted_indices[0], unsorted_indices=trii.unsorted_indices[0])
    mask = mask[0]
    # tdi  = tdi[0]

    return ten_1mm, ten_2mm, fod, brain, step, trid, trii, mask
    # return ten, fod, brain, step, trid, trii, mask
    # return ten, step, trid, trii, mask, tdi
