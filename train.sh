#!/bin/bash

#SBATCH --time=21-00:00:00
#SBATCH --constraint=A100
#SBATCH --mail-user=micah.bowles@postgrad.manchester.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=17
#SBATCH --job-name=F_MAE
#SBATCH --output=logs/%j.%x.out

# Source venv and call training
source /share/nas2_5/mbowles/venv/bin/activate
python /share/nas2_5/mbowles/mae/mae/train_timm.py
