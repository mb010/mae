#!/bin/bash

#SBATCH --time=7-00:00:00
#SBATCH --constraint=A100
#SBATCH --mail-user=micah.bowles@postgrad.manchester.ac.uk
#SBATCH --nodes=1
#SBATCH --mem=24G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=17
#SBATCH --job-name=F_MAE
#SBATCH --array=0-9
#SBATCH --output=logs/%A_%a.%x.out
#SBATCH --exclude=compute-0-4


EXPERIMENT_CONFIGS=(
    "fits_3_fft_first_pre.yml" #0 # Not currently used (small pretrained not available)
    "fits_2_fft_first_pre.yml" #1 # Not currently used (small pretrained not available)
    "fits_3_fft_first_scratch.yml" #2
    "fits_2_fft_first_scratch.yml" #3
    "fits_3_fft_first_scratch_noaug.yml" #4
    "fits_3_img_first_scratch.yml" #5 Pretrained version of this?
    "fits_3_fft_rgz_scratch.yml" #6
    "png_3_fft_first_scratch.yml" #7
    "png_3_img_first_scratch.yml" #8
    "png_3_fft_first_scratch_noaug.yml" #9
)

# List of options

EXPERIMENT_CONFIG=${EXPERIMENT_CONFIGS[$SLURM_ARRAY_TASK_ID]}
if [[ $EXPERIMENT_CONFIG == *'rgz'* ]]; then
    GLOBAL="global_rgzfits.yml"
else
    GLOBAL="global_firstfits.yml"
fi

# Accessing the option at the specified index
EXPERIMENT="${options[$index]}"

echo ">>> Starting call for: ${GLOBAL} and ${EXPERIMENT_CONFIG}"
# Source venv and call training
source /share/nas2_5/mbowles/venv/bin/activate
python -W ignore /share/nas2_5/mbowles/mae/mae/train_timm.py \
    $GLOBAL \
    $EXPERIMENT_CONFIG
