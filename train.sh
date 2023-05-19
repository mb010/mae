#!/bin/bash

#SBATCH --time=21-00:00:00
#SBATCH --constraint=A100
#SBATCH --mail-user=micah.bowles@postgrad.manchester.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=17
#SBATCH --job-name=F_MAE
#SBATCH --output=logs/%j.%x.out

# Move data to scratch for faster loading
if [ -d /state/partition1/fmae_data ]; then
    echo "/state/partition1/fmae_data already exists."
else
    echo "Copying data to /state/partition1/fmae_data"
    mkdir /state/partition1/fmae_data
    cp -r /share/nas2_5/mbowles/_data/FIRST /state/partition1/fmae_data/
    cp -r /share/nas2_5/mbowles/_data/MiraBest_FITS /state/partition1/fmae_data/
    cp -r /share/nas2_5/mbowles/_data/rgz_fits /state/partition1/fmae_data/
fi

# Source venv and call training
source /share/nas2_5/mbowles/venv/bin/activate
python /share/nas2_5/mbowles/mae/mae/train_timm.py

# Clean up scratch disk
echo "Removing /state/partition1/fmae_data/"
rm -r /state/partition1/fmae_data
