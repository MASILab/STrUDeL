# Prepare PyTorch Training
# Leon Cai
# MASI Lab
# July 21, 2022

# Set Up

import torch
import torch.nn as nn
from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space

from tqdm import tqdm
import numpy as np
import nibabel as nib
import os
import sys

from utils import streamline2network, len2mask, triinterp

# Go!

if __name__ == '__main__':

    tck_file = sys.argv[1]
    ref_file = sys.argv[2]
    batch_size = int(sys.argv[3])
    out_dir = sys.argv[4]

    print('prep_pt.py: Parsing inputs...')

    assert os.path.exists(tck_file), 'Input tck file does not exist. Aborting.'
    assert os.path.exists(tck_file), 'Input reference NIFTI file does not exist. Aborting.'
    assert batch_size > 0, 'Input batch size must be a positive integer. Aborting.'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    print('prep_pt.py: Input tck (and reference): {} ({})'.format(tck_file, ref_file))
    print('prep_pt.py: Batch size:                {}'.format(batch_size))
    print('prep_pt.py: Output Directory:          {}'.format(out_dir))

    # Load tractography

    print('prep_pt.py: Loading reference and streamlines in voxel space...')

    ref_nii = nib.load(ref_file)
    ref_img = ref_nii.get_fdata()

    tractogram = load_tractogram(tck_file, reference=ref_nii, to_space=Space.VOX)
    streamlines = tractogram.streamlines
    shuffle_idxs = np.linspace(0, len(streamlines)-1, len(streamlines)).astype(int)
    np.random.shuffle(shuffle_idxs)
    streamlines = streamlines[shuffle_idxs]
    num_batches = np.ceil(len(streamlines) / batch_size).astype(int)

    # Format streamlines

    for i, streamline_vox in tqdm(enumerate(streamlines), total=len(streamlines), desc='prep_pt.py: Formatting and saving streamlines...'):
        if i % batch_size == 0:
            b = i // batch_size
            if b > 0:
                torch.save(nn.utils.rnn.pad_sequence(streamlines_step, batch_first=False), os.path.join(out_dir, 'step_{:06}.pt'.format(b-1)))
                torch.save(nn.utils.rnn.pack_sequence(streamlines_trid, enforce_sorted=False), os.path.join(out_dir, 'trid_{:06}.pt'.format(b-1)))
                torch.save(nn.utils.rnn.pack_sequence(streamlines_trii, enforce_sorted=False), os.path.join(out_dir, 'trii_{:06}.pt'.format(b-1)))
                torch.save(torch.FloatTensor(len2mask(streamlines_len)), os.path.join(out_dir, 'mask_{:06}.pt'.format(b-1)))
            streamlines_step = []
            streamlines_trid = []
            streamlines_trii = []
            streamlines_len  = []
            streamlines_tdi  = []
        cut = np.random.randint(2, streamline_vox.shape[0]-2)
        streamline_trid, streamline_trii, streamline_step = streamline2network(streamline_vox, ref_img)
        streamline_tdi = triinterp(ref_img, streamline_trid, streamline_trii, fourth_dim=False)
        streamlines_step.append(torch.FloatTensor(streamline_step))
        streamlines_trid.append(torch.FloatTensor(streamline_trid))
        streamlines_trii.append(torch.LongTensor(streamline_trii))
        streamlines_len.append(streamline_step.shape[0])
    torch.save(nn.utils.rnn.pad_sequence(streamlines_step, batch_first=False), os.path.join(out_dir, 'step_{:06}.pt'.format(num_batches-1)))
    torch.save(nn.utils.rnn.pack_sequence(streamlines_trid, enforce_sorted=False), os.path.join(out_dir, 'trid_{:06}.pt'.format(num_batches-1)))
    torch.save(nn.utils.rnn.pack_sequence(streamlines_trii, enforce_sorted=False), os.path.join(out_dir, 'trii_{:06}.pt'.format(num_batches-1)))
    torch.save(torch.FloatTensor(len2mask(streamlines_len)), os.path.join(out_dir, 'mask_{:06}.pt'.format(num_batches-1)))
