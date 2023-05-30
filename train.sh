#!/bin/bash

#SBATCH --time=7-00:00:00
#SBATCH --constraint=A100
#SBATCH --mail-user=micah.bowles@postgrad.manchester.ac.uk
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=17
#SBATCH --job-name=FMAE
#SBATCH --output=logs/%j.%x.out
#SBATCH --exclude=compute-0-4
#SBATCH --signal=SIGUSR1@90

# Make NCCL errors legible
export NCCL_DEBUG=INFO

# Source venv and call training
source /share/nas2_5/mbowles/venv/bin/activate
python /share/nas2_5/mbowles/mae/mae/train_timm.py \
    global_firstfits.yml \
    fits_3_fft_first_scratch_noaug.yml
