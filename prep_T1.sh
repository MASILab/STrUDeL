#!/bin/bash

# Inputs:
# - T1.nii.gz

in_dir=$1
gpu=$2
atlas_dir=/home-local/dt1/code/pilot
slant_simg=/home-local/dt1/code/pilot/slant.simg
tractseg_simg=/home-local/dt1/code/pilot/tractSeg.simg
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$gpu

# Run SLANT:

echo "prep_T1.sh: Running SLANT..."
mkdir -p $in_dir/slant
singularity exec -e --bind /tmp:/tmp --bind $in_dir/T1.nii.gz:/INPUTS/T1.nii.gz --bind $in_dir/slant:/OUTPUTS --nv $slant_simg /extra/run_deep_brain_seg.sh
python group_slant.py $in_dir/slant/FinalResult/T1_seg.nii.gz $in_dir/T1_slant.nii.gz # T1_slant is one-hot encoded
# rm -r $in_dir/slant

# Generate mask:
# - T1_mask.nii.gz

echo "prep_T1.sh: Computing T1 mask..."
fslmaths $in_dir/T1_seg.nii.gz -div $in_dir/T1_seg.nii.gz -fillh $in_dir/T1_mask.nii.gz -odt int

# Bias correction
# - T1_N4.nii.gz

echo "prep_T1.sh: Bias correcting T1..."
N4BiasFieldCorrection -d 3 -i $in_dir/T1.nii.gz -x $in_dir/T1_mask.nii.gz -o $in_dir/T1_N4.nii.gz

# Generate tissue classes
# - T1_5tt.nii.gz

echo "prep_T1.sh: Computing 5tt classes..."
5ttgen fsl $in_dir/T1_N4.nii.gz $in_dir/T1_5tt.nii.gz -mask $in_dir/T1_mask.nii.gz -nocrop

# Generate seed map:
# - T1_seed.nii.gz

echo "prep_T1.sh: Computing seed mask..."
fslmaths $in_dir/T1_5tt.nii.gz -roi 0 -1 0 -1 0 -1 2 1 -bin -Tmax $in_dir/T1_seed.nii.gz -odt int
5tt2gmwmi -mask_in $in_dir/T1_mask.nii.gz $in_dir/T1_5tt.nii.gz $in_dir/T1_gmwmi.nii.gz

# Generate bundle priors:

echo "prep_T1.sh: Computing bundle priors..."
mkdir -p $in_dir/wm_learning
singularity run --contain -e --bind /tmp:/tmp --bind $in_dir/T1_N4.nii.gz:/INPUTS/T1_N4.nii.gz --bind $in_dir/wm_learning:/OUTPUTS --nv $tractseg_simg
fslmerge -t $in_dir/T1_tractseg.nii.gz $in_dir/wm_learning/tractSeg/orig/*.nii.gz
# rm -r $in_dir/wm_learning

# Register to MNI template:
# - T12mni_0GenericAffine.mat

echo "prep_T1.sh: Registering to MNI space at 1mm isotropic..."
antsRegistrationSyN.sh -d 3 -m $in_dir/T1_N4.nii.gz -f $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_1mm.nii.gz -t r -o $in_dir/T12mni_
mv $in_dir/T12mni_Warped.nii.gz $in_dir/T1_N4_mni_1mm.nii.gz
rm $in_dir/T12mni_InverseWarped.nii.gz

# Move data to MNI
# - T1_N4_mni_2mm.nii.gz
# - T1_mask_mni_2mm.nii.gz
# - T1_seed_mni_2mm.nii.gz
# - T1_5tt_mni_2mm.nii.gz

echo "prep_T1.sh: Moving images to MNI space at 1mm isotropic..."
antsApplyTransforms -d 3 -e 0 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_1mm.nii.gz -i $in_dir/T1_mask.nii.gz -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_mask_mni_1mm.nii.gz -n NearestNeighbor
antsApplyTransforms -d 3 -e 0 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_1mm.nii.gz -i $in_dir/T1_seed.nii.gz -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_seed_mni_1mm.nii.gz -n NearestNeighbor
antsApplyTransforms -d 3 -e 3 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_1mm.nii.gz -i $in_dir/T1_5tt.nii.gz  -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_5tt_mni_1mm.nii.gz  -n Linear
antsApplyTransforms -d 3 -e 0 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_1mm.nii.gz -i $in_dir/T1_gmwmi.nii.gz -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_gmwmi_mni_1mm.nii.gz -n Linear
antsApplyTransforms -d 3 -e 0 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_1mm.nii.gz -i $in_dir/T1_seg.nii.gz  -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_seg_mni_1mm.nii.gz  -n NearestNeighbor
# antsApplyTransforms -d 3 -e 3 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_1mm.nii.gz -i $in_dir/T1_tractseg.nii.gz  -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_tractseg_mni_1mm.nii.gz -n Linear
# antsApplyTransforms -d 3 -e 3 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_1mm.nii.gz -i $in_dir/T1_slant.nii.gz  -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_slant_mni_1mm.nii.gz -n NearestNeighbor

echo "prep_T1.sh: Moving images to MNI space at 2mm isotropic..."
antsApplyTransforms -d 3 -e 0 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_N4.nii.gz   -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_N4_mni_2mm.nii.gz   -n Linear
antsApplyTransforms -d 3 -e 0 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_mask.nii.gz -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_mask_mni_2mm.nii.gz -n NearestNeighbor
antsApplyTransforms -d 3 -e 0 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_seed.nii.gz -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_seed_mni_2mm.nii.gz -n NearestNeighbor
antsApplyTransforms -d 3 -e 3 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_5tt.nii.gz  -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_5tt_mni_2mm.nii.gz  -n Linear
antsApplyTransforms -d 3 -e 0 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_gmwmi.nii.gz -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_gmwmi_mni_2mm.nii.gz -n Linear
antsApplyTransforms -d 3 -e 0 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_seg.nii.gz  -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_seg_mni_2mm.nii.gz  -n NearestNeighbor
antsApplyTransforms -d 3 -e 3 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_tractseg.nii.gz  -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_tractseg_mni_2mm.nii.gz -n Linear
antsApplyTransforms -d 3 -e 3 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_slant.nii.gz  -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_slant_mni_2mm.nii.gz -n NearestNeighbor

# Warp rigid MNI to full template:

echo "prep_T1.sh: Non-linearly register rigidly registered image in MNI space to template at 1mm isotropic..."
antsRegistrationSyNQuick.sh -d 3 -m $in_dir/T1_N4_mni_1mm.nii.gz -f $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_1mm.nii.gz -t s -o $in_dir/mni2warp_
echo "prep_T1.sh: Warp template positional encodings to rigidly registered image in MNI space at 2mm isotropic..."
antsApplyTransforms -d 3 -e 3 -r $in_dir/T1_N4_mni_2mm.nii.gz -i $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm_posenc16.nii.gz -t [$in_dir/mni2warp_0GenericAffine.mat,1] -t $in_dir/mni2warp_1InverseWarp.nii.gz -o $in_dir/T1_posenc16_mni_2mm.nii.gz -n Linear

# Wrap up

echo "prep_T1.sh: Done!"
