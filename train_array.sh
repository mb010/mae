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


EXPERIMENT_CONFIGS=(
    "fits_3_fft_first_pre.yml"
    "fits_2_fft_first_pre.yml"
    "fits_3_fft_first_scratch.yml"
    "fits_2_fft_first_scratch.yml"
    "fits_3_fft_first_scratch_noaug.yml"
    "fits_3_img_first_scratch.yml" # Pretrained version of this?
    "fits_3_fft_rgz_scratch.yml"
    "png_3_fft_first_scratch.yml"
    "png_3_img_first_scratch.yml"
    "png_3_fft_first_scratch_noaug.yml"
)

# List of options

EXPERIMENT_CONFIG=${EXPERIMENT_CONFIGS[$SLURM_ARRAY_TASK_ID]}
if [[ EXPERIMENT_CONFIG == *"rgz"* ]]; then
    GLOBAL="global_rgzfits.yml"
else
    GLOBAL="gloal_firstfits.yml"
fi

# Accessing the option at the specified index
EXPERIMENT="${options[$index]}"

# Move data to scratch for faster loading
echo ">>> Copying data to /state/partition1/fmae_data"
mkdir -p /state/partition1/fmae_data
rsync -urltv /share/nas2_5/mbowles/_data/FIRST /state/partition1/fmae_data/
rsync -urltv /share/nas2_5/mbowles/_data/MiraBest_FITS /state/partition1/fmae_data/
rsync -urltv /share/nas2_5/mbowles/_data/rgz_fits /state/partition1/fmae_data/

echo ">>> Starting call for: ${GLOBAL} and ${EXPERIMENT_CONFIG}"
# Source venv and call training
source /share/nas2_5/mbowles/venv/bin/activate
python /share/nas2_5/mbowles/mae/mae/train_timm.py \
    $GLOBAL \
    $EXPERIMENT_CONFIG


# Clean up scratch disk
echo ">>> Removing /state/partition1/fmae_data/"
rm -r /state/partition1/fmae_data
