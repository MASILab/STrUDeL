# STrUDeL: Streamline Tractography Utilities for Deep Learning

## Authors and Reference

If you use this code base in your work, please cite the following papers:

[Leon Y. Cai](mailto:leon.y.cai@vanderbilt.edu), Ho Hin Lee, Nancy R. Newlin, Michael E. Kim, Daniel Moyer, Francois Rheault, Kurt G. Schilling, and Bennett A. Landman. Implementation considerations for deep learning with diffusion MRI streamline tractography. *In Submission*. 2023.

Leon Y. Cai, Ho Hin Lee, Nancy R. Newlin, Cailey I. Kerley, Praitayini Kanakaraj, Qi Yang, Graham W. Johnson, Daniel Moyer, Kurt G. Schilling, Francois Rheault, and Bennett A. Landman. [Convolutional-recurrent neural networks approximate diffusion tractography from T1-weighted MRI and associated anatomical context](https://www.biorxiv.org/content/10.1101/2023.02.25.530046v2). Proceedings of Machine Learning Reseach. In press. 2023.

If you are interested in propagating streamlines directly on T1-weighted MRI using a Singularity container without reimplementing or retraining, please visit [github.com/MASILab/cornn_tractography](https://github.com/MASILab/cornn_tractography).

This work is a product of the [Medical-image Analysis and Statistical Interpretation (MASI) Lab](https://my.vanderbilt.edu/masi) at Vanderbilt University in Nashville, TN, USA.

## Tutorial

The following sections outline the necessary steps to both train and test a recurrent neural network model for diffusion MRI streamline tractography on one publicly available dataset using the the STrUDeL framework. 

### Set up the code base

Clone the STrUDeL repository and install the dependencies.

```
git clone https://github.com/MASILab/STrUDeL.git
cd STrUDeL
bash install.sh
source venv/bin/activate
```

`install.sh` may need to be modified depending on your system. This tutorial was tested on an Ubuntu 20.04 x86_64 machine with 64GB of CPU RAM and an NVIDIA RTX Quadro 5000 with 16GB of GPU RAM running CUDA 11.6. This tutorial also requires the following libraries:

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
wget -O dwmri.nii.gz https://openneuro.org/crn/datasets/ds003416/snapshots/2.0.2/files/derivatives:prequal-v1.0.0:sub-cIIs00:ses-s1Ax1:dwi:sub-cIIs00_ses-s1Ax1_acq-b2000n96r25x25x25peAPP_run-107_dwi.nii.gz
wget -O dwmri.bvec https://openneuro.org/crn/datasets/ds003416/snapshots/2.0.2/files/derivatives:prequal-v1.0.0:sub-cIIs00:ses-s1Ax1:dwi:sub-cIIs00_ses-s1Ax1_acq-b2000n96r25x25x25peAPP_run-107_dwi.bvec
wget -O dwmri.bval https://openneuro.org/crn/datasets/ds003416/snapshots/2.0.2/files/derivatives:prequal-v1.0.0:sub-cIIs00:ses-s1Ax1:dwi:sub-cIIs00_ses-s1Ax1_acq-b2000n96r25x25x25peAPP_run-107_dwi.bval
wget -O T1.nii.gz https://openneuro.org/crn/datasets/ds003416/snapshots/2.0.2/files/sub-cIIs00:ses-s1Ax1:anat:sub-cIIs00_ses-s1Ax1_acq-r10x10x10_T1w.nii.gz
cd ..
```

### Preprocess and format the data

Compute the necessary masks and tissue-type segmentations from the T1-weighted MRI for anatomically constrained tractography

```
bash prep_T1.sh data/
```

Fit the diffusion data to a fiber orientation distribution (FOD) model and perform SDStream tractography to obtain the ground truth tractograms

```
bash prep_dwmri.sh data/
```

Format the data for deep learning with a PyTorch DataLoader.

```
python prep_pt.py data/
echo ${pwd}/data > data.txt
```

### Train the model

Train the model! 

```
python train.py data.txt
```

This script uses functions defined in `data.py`, `modules.py`, and `utils.py`. These functions contain the majority of the logic outlined in the [STrUDeL paper](#authors-and-reference).

### Test the model

Test the model!

```
python generate.py data/fod.nii.gz --pretrained
```

Use the `--pretrained` flag to use the model characterized in the [STrUDeL paper](#authors-and-reference). Otherwise, this script will use the weights from the previous step.

### Visualize the results

The tractograms, both the ground truth and the predicted, can be visualized in `mrview`, provided by MRTrix3.

