# STrUDeL: Streamline Tractography Utilities for Deep Learning

This repository contains implementations for training recurrent neural network models for streamline tractography on diffusion MRI. The source code is provided without containerized implementation to facilitate further development in this field. We additionally provide a tutorial walking through a toy example for how the function in this repository may be used. If you are interested in propagating streamlines directly on T1-weighted MRI using a Singularity container without reimplementing or retraining, please visit [github.com/MASILab/cornn_tractography](https://github.com/MASILab/cornn_tractography).

## Authors and Reference

If you use this code base in your work, please cite the following papers:

* **The STrUDeL paper:** [Leon Y. Cai](mailto:leon.y.cai@vanderbilt.edu), Ho Hin Lee, Nancy R. Newlin, Michael E. Kim, Daniel Moyer, Francois Rheault, Kurt G. Schilling, and Bennett A. Landman. [Implementation considerations for deep learning with diffusion MRI streamline tractography](https://www.biorxiv.org/content/10.1101/2023.04.03.535465v1). Proceedings of Machine Learning Reseach. In press. 2023.

* **The CoRNN paper:** Leon Y. Cai, Ho Hin Lee, Nancy R. Newlin, Cailey I. Kerley, Praitayini Kanakaraj, Qi Yang, Graham W. Johnson, Daniel Moyer, Kurt G. Schilling, Francois Rheault, and Bennett A. Landman. [Convolutional-recurrent neural networks approximate diffusion tractography from T1-weighted MRI and associated anatomical context](https://www.biorxiv.org/content/10.1101/2023.02.25.530046v2). Proceedings of Machine Learning Reseach. In press. 2023.

This work is a product of the [Medical-image Analysis and Statistical Interpretation (MASI) Lab](https://my.vanderbilt.edu/masi) at Vanderbilt University in Nashville, TN, USA.

## Tutorial

The following sections outline the necessary steps to both train and test a recurrent neural network model for diffusion MRI streamline tractography on one publicly available dataset using the the STrUDeL framework. 

### Set up the code base

Clone the STrUDeL repository and install the dependencies.

```
git clone https://github.com/MASILab/STrUDeL.git
cd STrUDeL
python3 -m venv venv
source venv/bin/activate
pip install dipy==1.5.0
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install tensorboard==2.10.0
```

Note: The `torch` installation may need to be modified depending on your system. This tutorial was tested on an Ubuntu 20.04 x86_64 machine with 64GB of CPU RAM and an NVIDIA RTX Quadro 5000 with 16GB of GPU RAM running CUDA 11.6. 

This tutorial also requires the following libraries:

* [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation#Installing_FSL)
* [ANTs](https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS)
* [MRTrix3](https://www.mrtrix.org/download/)
* [ScilPy](https://github.com/scilus/scilpy)

### Get the data

Download a publicly available, preprocessed diffusion and T1-weighted MRI dataset from OpenNeuro. This dataset includes imaging from multiple sessions, sites, and scanners and can be cited as follows:

Leon Y. Cai, Qi Yang, Praitayini Kanakaraj, Vishwesh Nath, Allen T. Newton, Heidi A. Edmonson, Jeffrey Luci, Benjamin N. Conrad, Gavin R. Price, Colin B. Hansen, Cailey I. Kerley, Karthik Ramadass, Fang-Cheng Yeh, Hakmook Kang, Eleftherios Garyfallidis, Maxime Descoteaux, Francois Rheault, Kurt G. Schilling, and Bennett A. Landman. MASiVar: Multisite, Multiscanner, and Multisubject Acquisitions for Studying Variability in Diffusion Weighted Magnetic Resonance Imaging. [Magnetic Resonance in Medicine](https://doi.org/10.1002/mrm.28926), 2021.

```
mkdir data
cd data
wget -O dmri.nii.gz https://openneuro.org/crn/datasets/ds003416/snapshots/2.0.2/files/derivatives:prequal-v1.0.0:sub-cIIs00:ses-s1Ax1:dwi:sub-cIIs00_ses-s1Ax1_acq-b2000n96r25x25x25peAPP_run-107_dwi.nii.gz
wget -O dmri.bvec https://openneuro.org/crn/datasets/ds003416/snapshots/2.0.2/files/derivatives:prequal-v1.0.0:sub-cIIs00:ses-s1Ax1:dwi:sub-cIIs00_ses-s1Ax1_acq-b2000n96r25x25x25peAPP_run-107_dwi.bvec
wget -O dmri.bval https://openneuro.org/crn/datasets/ds003416/snapshots/2.0.2/files/derivatives:prequal-v1.0.0:sub-cIIs00:ses-s1Ax1:dwi:sub-cIIs00_ses-s1Ax1_acq-b2000n96r25x25x25peAPP_run-107_dwi.bval
wget -O T1.nii.gz https://openneuro.org/crn/datasets/ds003416/snapshots/2.0.2/files/sub-cIIs00:ses-s1Ax1:anat:sub-cIIs00_ses-s1Ax1_acq-r10x10x10_T1w.nii.gz
```

### Prepare T1-weighted MRI

Generate the masks from T1-weighted MRI need for anatomically constrained tractography with dMRI.

1. Extract the brain and convert it to a binary mask with FSL: `T1_mask.nii.gz`
```
bet T1.nii.gz T1_bet.nii.gz -f 0.25 -R
fslmaths T1_bet.nii.gz -div T1_bet.nii.gz -fillh T1_mask.nii.gz -odt int
```
2. Perform N4 bias correction on the T1 with ANTs: `T1_N4.nii.gz`
```
N4BiasFieldCorrection -d 3 -i T1.nii.gz -x T1_mask.nii.gz -o T1_N4.nii.gz
```
3. Compute a tissue-type mask with MRTrix3 and FSL: `T1_5tt.nii.gz`
```
5ttgen fsl T1_N4.nii.gz T1_5tt.nii.gz -mask T1_mask.nii.gz -nocrop
```
4. Compute a WM seeding mask from the tissue-type mask with FSL: `T1_seed.nii.gz`
```
fslmaths T1_5tt.nii.gz -roi 0 -1 0 -1 0 -1 2 1 -bin -Tmax T1_seed.nii.gz -odt int
```

### Prepare dMRI

Fit the diffusion data to a fiber orientation distribution (FOD) model and perform SDStream tractography to obtain a ground truth tractogram

1. Compute the average b0 from the diffusion data with MRTrix3: `dmri_b0.nii.gz`
```
dwiextract dmri.nii.gz -fslgrad dmri.bvec dmri.bval - -bzero | mrmath - mean dmri_b0.nii.gz -axis 3
```
2. Compute a rigid transform between the average b0 and T1 with ANTs: `dmri2T1_0GenericAffine.mat`
```
antsRegistrationSyN.sh -d 3 -m dmri_b0.nii.gz -f T1_N4.nii.gz -t r -o dmri2T1_
rm dmri2T1_Warped.nii.gz dmri2T1_InverseWarped.nii.gz # clean up
```
3. Move the structural masks to diffusion space with ANTs: `dmri_mask.nii.gz`, `dmri_seed.nii.gz`, `dmri_5tt.nii.gz`
```
antsApplyTransforms -d 3 -e 0 -r dmri_b0.nii.gz -i T1_mask.nii.gz -t [dmri2T1_0GenericAffine.mat,1] -o dmri_mask.nii.gz -n NearestNeighbor
antsApplyTransforms -d 3 -e 0 -r dmri_b0.nii.gz -i T1_seed.nii.gz -t [dmri2T1_0GenericAffine.mat,1] -o dmri_seed.nii.gz -n NearestNeighbor
antsApplyTransforms -d 3 -e 3 -r dmri_b0.nii.gz -i T1_5tt.nii.gz  -t [dmri2T1_0GenericAffine.mat,1] -o dmri_5tt.nii.gz  -n Linear
```
4. Fit the diffusion data with FODs with MRTrix3: `dmri_fod.nii.gz`
```
dwi2response tournier dmri.nii.gz dmri_tournier.txt -fslgrad dmri.bvec dmri.bval -mask dmri_mask.nii.gz
dwi2fod csd dmri.nii.gz dmri_tournier.txt dmri_fod.nii.gz -fslgrad dmri.bvec dmri.bval -mask dmri_mask.nii.gz
```
5. Perform SDStream tractography with MRTrix3: `dmri_sdstream.tck`
```
tckgen dmri_fod.nii.gz dmri_sdstream.tck -algorithm SD_Stream -select 1000000 -step 1 -seed_image dmri_seed.nii.gz -mask dmri_mask.nii.gz -minlength 50 -maxlength 250 -act dmri_5tt.nii.gz
```

### Move the data to common space for training and testing

This step is useful if this framework is to be extended to more than one dataset that may be oriented differently or to use the pretrained model which expects data in this common space, but is not strictly required for training and evaluation on only one participant's data.

1. Compute a rigid transformation to a common MNI space at 1mm isotropic resolution with ANTs: `T12mni_0GenericAffine.mat`
```
antsRegistrationSyN.sh -d 3 -m T1_N4.nii.gz -f ../mni/mni_icbm152_t1_tal_nlin_asym_09c_1mm.nii.gz -t r -o T12mni_
rm T12mni_Warped.nii.gz T12mni_InverseWarped.nii.gz # clean up
```
2. Apply that transformation to the T1 and masks at 2mm isotropic resolution with ANTs: `T1_N4_mni_2mm.nii.gz`, `T1_mask_mni_2mm.nii.gz`, `T1_5tt_mni_2mm.nii.gz`, `T1_seed_mni_2mm.nii.gz`
```
antsApplyTransforms -d 3 -e 0 -r ../mni/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i T1_N4.nii.gz   -t T12mni_0GenericAffine.mat -o T1_N4_mni_2mm.nii.gz  -n Linear
antsApplyTransforms -d 3 -e 0 -r ../mni/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i T1_mask.nii.gz -t T12mni_0GenericAffine.mat -o T1_mask_mni_2mm.nii.gz -n NearestNeighbor
antsApplyTransforms -d 3 -e 3 -r ../mni/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i T1_5tt.nii.gz  -t T12mni_0GenericAffine.mat -o T1_5tt_mni_2mm.nii.gz  -n Linear
antsApplyTransforms -d 3 -e 0 -r ../mni/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i T1_seed.nii.gz -t T12mni_0GenericAffine.mat -o T1_seed_mni_2mm.nii.gz -n NearestNeighbor
```
3. Apply that transformation to the FODs with reorientation, first walking through T1 space with MRTrix3 and ANTs: `T1_fod_mni_2mm.nii.gz`
```
ConvertTransformFile 3 dmri2T1_0GenericAffine.mat dmri2T1_0GenericAffine.txt
transformconvert dmri2T1_0GenericAffine.txt itk_import dmri2T1_0GenericAffine.txt -force
ConvertTransformFile 3 T12mni_0GenericAffine.mat T12mni_0GenericAffine.txt
transformconvert T12mni_0GenericAffine.txt itk_import T12mni_0GenericAffine.txt -force
mrtransform -linear dmri2T1_0GenericAffine.txt -modulate fod -reorient_fod true dmri_fod.nii.gz - | mrtransform -linear T12mni_0GenericAffine.txt -interp linear -template T1_N4_mni_2mm.nii.gz -stride T1_N4_mni_2mm.nii.gz -modulate fod -reorient_fod true - T1_fod_mni_2mm.nii.gz
```
4. Apply that transformation to the tractogram, first walking through T1 space with ScilPy: `T1_sdstream.tck`, `T1_sdstream_mni_2mm.tck`
```
scil_apply_transform_to_tractogram.py --reference dmri_b0.nii.gz dmri_sdstream.tck T1_N4.nii.gz         dmri2T1_0GenericAffine.mat T1_sdstream.tck         --inverse --remove_invalid
scil_apply_transform_to_tractogram.py --reference T1_N4.nii.gz   T1_sdstream.tck   T1_N4_mni_2mm.nii.gz T12mni_0GenericAffine.mat  T1_sdstream_mni_2mm.tck --inverse --remove_invalid
```

### Format the data for deep learning

```
python ../prep_pt.py T1_sdstream_mni_2mm.tck T1_N4_mni_2mm.nii.gz 1000 pt
```

As described in the [STrUDeL paper](#authors-and-reference), this script takes a tractogram with 1 million streamlines, `T1_sdstream_mni_2mm.tck`, defined in the same space as `T1_N4_mni_2mm.nii.gz`, and splits the streamlines up into batches of size `K = 1000`. It does so using the following functions:

* `dipy.io.streamline.load_tractogram()` loads all 1 million streamlines into a Nibabel array sequence in voxel space (voxel space is critical to ensure efficient querying of the voxel grid)
* For each streamline, we compute the ground truth label, `∆x` in the paper, for each point using `utils.vox2step()`. `step` is the prefix.
* For each streamline, we convert each point to 11-dimensional trilinear interpolation format, `c` in the paper, using `utils.vox2trid()` for the first three elements and `utils.vox2trii()` for the last 8. `trid` and `trii` are the prefixes, respectively.
* `utils.vox2step()`, `utils.vox2trid()`, and `utils.vox2trii()` are encapsulated in `utils.streamline2network()` which also removes the last point of each streamline so that the data and labels are the same length

These outputs are then saved in a list for all streamlines in the batch and subsequently saved as packed tensors (`trii` and `trid`) or padded tensors (`step`) along with a padded tensor allowing the padding to be ignored during loss computation with prefix `mask` computed with `utils.len2mask()`. These data for each batch are then saved in the `pt` folder as `<prefix>_<batch_number>.pt`.

* `torch.nn.utils.rnn.pad_sequence()` facilitates padding
* `torch.nn.utils.rnn.pack_sequence()` facilitates packing

During writing of this tutorial, this step took about an hour to process all 1 million streamlines and produced about 10GB worth of files.

### Train the model

First, write the `data` folder we've been operating in into a text file so the PyTorch data loader knows where to look during training for the prepared labels and streamline batches. Note the above preparation steps can be repeated for other data to enable training on multiple datasets.
```
echo $(pwd) > ../data.txt
```
Then, train the model with the data recorded in that text file for 500 epochs:
```
cd ..
python train.py data.txt 1000000 1000 tensorboard/ 500
```

This script uses functions defined in `data.py` and `modules.py`. These functions contain the majority of the logic outlined in the [STrUDeL paper](#authors-and-reference). 

In brief, each epoch the data loader will return the FOD voxel grid along with a randomly selected batch of `K = 1000` streamlines from the possible `1000000` prepared above and formulated for trilinear interpolation with `modules.TrilinearInterpolator.forward()` (`trid` and `trii`), as returned in `data.STrUDeLDataset.__getitem__()`. These are then run through the model in the forward pass, including the interpolation step, defined in `modules.RecurrentModel.forward()` to return the predicted `step_pred`, which is then used with the label `step` in a pass of `modules.CosineLoss.forward()` to compute the loss. The losses are backpropagated each batch and the average loss per epoch is saved to the `tensorboard/` folder for viewing, if desired.

The weights from this step are saved in `weights_tutorial.pt`

### Test the model

Test the model: 
```
model=tutorial
python generate.py data/T1_fod_mni_2mm.nii.gz data/T1_seed_mni_2mm.nii.gz data/T1_mask_mni_2mm.nii.gz data/T1_5tt_mni_2mm.nii.gz weights_${model}.pt data/T1_${model}_mni_2mm.tck
```
Set `model` to `tutorial` to use the weights from the previous step. Set it to `pretrained` to use the model characterized in the [STrUDeL paper](#authors-and-reference). This script will output tractograms to either `data/T1_tutorial_mni_2mm.tck` or `data/T1_pretrained_mni_2mm.tck`.

This script implements the tracking design described in the [CoRNN paper](#authors-and-reference), using a min/max length of 5/250mm, a maximum step angle of 60°, and a step size of 1mm with anatomical constraints analogous to those used by `tckgen` in MRTrix3 and bidirectional tracking with random seeding in the WM. It also uses the batch-wise implementation described in the [STrUDeL paper](#authors-and-reference). The following is a description of the functions used by this script:

* `generate.tri2act()` implements the logic for anatomically constrained stopping criteria given the current and previous steps in the streamlines. This function uses the same trilinear interpolation framework, implemnted in `utils.triinterp()`.
* `generate.img2seeds()` implements random seeding in WM
* `generate.seeds2streamlines()` implements the primary tracking steps, propagating in one direction from the seeded locations
* `generate.streamlines2reverse()` implements preparation for tracking in the reverse direction, flipping the streamlines and performing a forward pass to obtain new "seeds", which can then be passed to `generate.seeds2streamlines()` again.

### Visualize the results

The tractograms, both the ground truth and the predicted, can be visualized in `mrview` provided by MRTrix3 or `MI-Brain` by Imeka. Note that the FOV of the dMRI image does not include the cerebellum, so the fibers there are largely missing or poorly defined.

![example_tractograms](https://github.com/MASILab/STrUDeL/blob/master/example.png?raw=true)
