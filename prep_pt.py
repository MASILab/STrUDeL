# Prepare PyTorch Training
# Leon Cai
# MASI Lab
# July 21, 2022

# Set Up

import torch
import torch.nn as nn
from dipy.io.streamline import load_tractogram, save_tractogram, StatefulTractogram
from dipy.io.stateful_tractogram import Space

from tqdm import tqdm
import numpy as np
import nibabel as nib
import scipy
import os
import sys

from utils import streamline2network, len2mask, triinterp

# Go!

if __name__ == '__main__':

    in_dir = sys.argv[1]
    num_streamlines = sys.argv[2]
    batch_size = sys.argv[3]

    # Parse inputs

    print('prep_pt.py: Parsing inputs...')

    assert os.path.exists(in_dir), 'Input directory does not exist.'

    trk_file = os.path.join(in_dir, 'T1_test50to250_mni_2mm.trk')
    t1_file = os.path.join(in_dir, 'T1_N4_mni_2mm.nii.gz')
    act_file = os.path.join(in_dir, 'T1_5tt_mni_2mm.nii.gz')
    mask_file = os.path.join(in_dir, 'T1_mask_mni_2mm.nii.gz')
    tdi_file = os.path.join(in_dir, 'T1_gmwmi_mni_2mm_tdi.nii.gz')
    fod_file = os.path.join(in_dir, 'dwmri_fod_mni_2mm_trix.nii.gz')
    sf_file = os.path.join(in_dir, 'dwmri_sf_mni_2mm_trix.nii.gz')
    tseg_file = os.path.join(in_dir, 'T1_tractseg_mni_2mm.nii.gz')
    slant_file = os.path.join(in_dir, 'T1_slant_mni_2mm.nii.gz')

    # assert os.path.exists(trk_file) and os.path.exists(t1_file) and os.path.exists(act_file) and os.path.exists(mask_file), 'Input directory missing files.'

    out_dir = os.path.join(in_dir, 'dt1_test50to250')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    num_streamlines = int(num_streamlines)
    batch_size = int(batch_size)
    num_batches = np.ceil(num_streamlines / batch_size).astype(int)

    print('prep_pt.py: Input Directory:   {}'.format(in_dir))
    print('prep_pt.py: Number of Batches: {}'.format(num_batches))
    print('prep_pt.py: Output Directory:  {}'.format(out_dir))

    # Load imaging

    print('prep_pt.py: Loading imaging...')

    t1_img = nib.load(t1_file).get_fdata()
    mask_img = nib.load(mask_file).get_fdata().astype(bool)
    act_img = nib.load(act_file).get_fdata()[:, :, :, :-1]
    tdi_img = nib.load(tdi_file).get_fdata()
    fod_img = nib.load(fod_file).get_fdata()
    sf_img = nib.load(sf_file).get_fdata()
    tseg_img = nib.load(tseg_file).get_fdata()
    slant_img = nib.load(slant_file).get_fdata()

    # Format imaging

    print('prep_pt.py: Formatting and saving imaging...')

    t1_ten = torch.FloatTensor(np.expand_dims(t1_img / np.median(t1_img[mask_img]), axis=(0, 1)))
    act_ten = torch.FloatTensor(np.expand_dims(np.transpose(act_img, axes=(3, 0, 1, 2)), axis=0))
    ten = torch.cat((t1_ten, act_ten), dim=1)
    torch.save(ten, os.path.join(out_dir, 'ten.pt'))

    fod_ten = torch.FloatTensor(np.expand_dims(np.transpose(fod_img, axes=(3, 0, 1, 2)), axis=0))
    torch.save(fod_ten, os.path.join(out_dir, 'fod.pt'))

    sf_ten = torch.FloatTensor(np.expand_dims(np.transpose(sf_img, axes=(3, 0, 1, 2)), axis=0))
    torch.save(sf_ten, os.path.join(out_dir, 'sf.pt'))

    mask_ten = torch.FloatTensor(np.expand_dims(mask_img, axis=(0, 1)))
    torch.save(mask_ten, os.path.join(out_dir, 'mask.pt'))

    tseg_ten = torch.FloatTensor(np.expand_dims(np.transpose(tseg_img, axes=(3, 0, 1, 2)), axis=0))
    torch.save(tseg_ten, os.path.join(out_dir, 'tseg.pt'))

    slant_ten = torch.FloatTensor(np.expand_dims(np.transpose(slant_img, axes=(3, 0, 1, 2)), axis=0))
    torch.save(slant_ten, os.path.join(out_dir, 'slant.pt'))

    # Load tractography

    print('prep_pt.py: Loading tractography...')

    tractogram = load_tractogram(trk_file, reference='same', to_space=Space.VOX)
    streamlines = tractogram.streamlines
    shuffle_idxs = np.linspace(0, len(streamlines)-1, len(streamlines)).astype(int)
    np.random.shuffle(shuffle_idxs)
    streamlines = streamlines[shuffle_idxs]

    # Compute weight function

    print('prep_pt.py: Computing TDI weights function...')

    tdi_mask = tdi_img > 0
    tdi_thres = np.percentile(tdi_img[tdi_mask], 99)
    tdi_img[tdi_img > tdi_thres] = tdi_thres

    tdi_streamlines = streamlines[:int(np.sum(tdi_mask)/10)]
    tdi_streamlines_trid, tdi_streamlines_trii, _ = streamline2network(np.concatenate(tdi_streamlines, axis=0), t1_img)
    tdi_streamlines_samples = triinterp(tdi_img, tdi_streamlines_trid, tdi_streamlines_trii, fourth_dim=False)
    tdi_voxels_samples = tdi_img[tdi_mask]

    tdi_streamlines_pdf, tdi_weights_edges = np.histogram(tdi_streamlines_samples, bins=99, range=(0, tdi_thres), density=True)
    tdi_voxels_pdf, _ = np.histogram(tdi_voxels_samples, bins=99, range=(0, tdi_thres), density=True)    
    tdi_streamlines2voxels = np.concatenate((tdi_voxels_pdf / tdi_streamlines_pdf, [0]), axis=0)
    tdi_weights = scipy.signal.medfilt(tdi_streamlines2voxels, 3)
    tdi_weights[0] = tdi_streamlines2voxels[0]
    tdi_weights[-1] = tdi_streamlines2voxels[-1]
    
    tdi_weights_fxn = scipy.interpolate.interp1d(tdi_weights_edges, tdi_streamlines2voxels, kind='linear', assume_sorted=True)
    torch.save(tdi_weights_fxn, os.path.join(out_dir, 'tdi_weights_fxn.pt'))

    # Format streamlines

    for i, streamline_vox in tqdm(enumerate(streamlines), total=len(streamlines), desc='prep_pt.py: Formatting and saving tractography...'):
        if i % batch_size == 0:
            b = i // batch_size
            if b > 0:
                torch.save(nn.utils.rnn.pad_sequence(streamlines_step, batch_first=False), os.path.join(out_dir, 'step_{:06}.pt'.format(b-1)))
                torch.save(nn.utils.rnn.pack_sequence(streamlines_trid, enforce_sorted=False), os.path.join(out_dir, 'trid_{:06}.pt'.format(b-1)))
                torch.save(nn.utils.rnn.pack_sequence(streamlines_trii, enforce_sorted=False), os.path.join(out_dir, 'trii_{:06}.pt'.format(b-1)))
                torch.save(torch.FloatTensor(len2mask(streamlines_len)), os.path.join(out_dir, 'mask_{:06}.pt'.format(b-1)))
                torch.save(nn.utils.rnn.pad_sequence(streamlines_tdi, batch_first=False), os.path.join(out_dir, 'tdi_{:06}.pt'.format(b-1)))
            streamlines_step = []
            streamlines_trid = []
            streamlines_trii = []
            streamlines_len  = []
            streamlines_tdi  = []
        cut = np.random.randint(2, streamline_vox.shape[0]-2)
        for streamline_vox_seg in [streamline_vox]: # [streamline_vox, np.flip(streamline_vox, axis=0)]: # , np.flip(streamline_vox[:cut, :], axis=0), streamline_vox[cut:, :]]:
            streamline_trid, streamline_trii, streamline_step = streamline2network(streamline_vox_seg, t1_img)
            streamline_tdi = triinterp(tdi_img, streamline_trid, streamline_trii, fourth_dim=False)
            streamlines_step.append(torch.FloatTensor(streamline_step))
            streamlines_trid.append(torch.FloatTensor(streamline_trid))
            streamlines_trii.append(torch.LongTensor(streamline_trii))
            streamlines_len.append(streamline_step.shape[0])
            streamlines_tdi.append(torch.FloatTensor(streamline_tdi))
    torch.save(nn.utils.rnn.pad_sequence(streamlines_step, batch_first=False), os.path.join(out_dir, 'step_{:06}.pt'.format(num_batches-1)))
    torch.save(nn.utils.rnn.pack_sequence(streamlines_trid, enforce_sorted=False), os.path.join(out_dir, 'trid_{:06}.pt'.format(num_batches-1)))
    torch.save(nn.utils.rnn.pack_sequence(streamlines_trii, enforce_sorted=False), os.path.join(out_dir, 'trii_{:06}.pt'.format(num_batches-1)))
    torch.save(torch.FloatTensor(len2mask(streamlines_len)), os.path.join(out_dir, 'mask_{:06}.pt'.format(num_batches-1)))
    torch.save(nn.utils.rnn.pad_sequence(streamlines_tdi, batch_first=False), os.path.join(out_dir, 'tdi_{:06}.pt'.format(num_batches-1)))
