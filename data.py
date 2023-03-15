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

class STrUDeLDataset(Dataset):

    def __init__(self, data_dirs, num_batches):

        super(STrUDeLDataset, self).__init__()
        self.data_dirs = data_dirs
        self.num_batches = num_batches

    def __getitem__(self, index):

        data_dir = self.data_dirs[index]

        # Read in FOD

        fod_file = os.path.join(data_dir, 'T1_fod_mni_2mm.nii.gz')
        fod_img  = nib.load(fod_file).get_fdata()
        fod      = torch.FloatTensor(np.expand_dims(np.transpose(fod_img, axes=(3, 0, 1, 2)), axis=0))

        # Read in a random streamline batch

        pt_dir = os.path.join(data_dir, 'pt')
        b = np.random.randint(0, self.num_batches)
        step = torch.load(os.path.join(pt_dir, 'step_{:06}.pt'.format(b)))
        trid = torch.load(os.path.join(pt_dir, 'trid_{:06}.pt'.format(b)))
        trii = torch.load(os.path.join(pt_dir, 'trii_{:06}.pt'.format(b)))
        mask = torch.load(os.path.join(pt_dir, 'mask_{:06}.pt'.format(b)))

        # Return data
        # Note: These will be wrapped in an arbitrary batch dimension due to PyTorch DataLoader behavior. Use the unload() function to remove this.

        return fod, step, trid, trii, mask

    def __len__(self):

        return len(self.data_dirs)

def unload(fod, step, trid, trii, mask):

    fod = fod[0]
    step = step[0]
    trid = nn.utils.rnn.PackedSequence(trid.data[0], batch_sizes=trid.batch_sizes[0], sorted_indices=trid.sorted_indices[0], unsorted_indices=trid.unsorted_indices[0])
    trii = nn.utils.rnn.PackedSequence(trii.data[0], batch_sizes=trii.batch_sizes[0], sorted_indices=trii.sorted_indices[0], unsorted_indices=trii.unsorted_indices[0])
    mask = mask[0]

    return fod, step, trid, trii, mask
