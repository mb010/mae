#!/bin/bash

#SBATCH --time=21-00:00:00
#SBATCH --constraint=A100
#SBATCH --mail-user=micah.bowles@postgrad.manchester.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=17
#SBATCH --job-name=ftFMAE
#SBATCH --output=logs/%j.%x.out


START=`date +%s`

# Source venv and call training
source /share/nas2_5/mbowles/venv/bin/activate
python /share/nas2_5/mbowles/mae/mae/finetuning.py \
    finetune.yml

END=`date +%s`
RUNTIME=$((END-START))
echo ">>> In total this run took ${RUNTIME}s."
