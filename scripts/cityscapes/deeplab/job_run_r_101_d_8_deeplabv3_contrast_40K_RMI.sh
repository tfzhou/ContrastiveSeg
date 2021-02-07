#!/bin/bash
#SBATCH -n 40
#SBATCH --output=log/%j-deeplabv3-contrast-40k-rmi.out
#SBATCH --gres=gpu:v100:4
#SBATCH --time=48:00:00

source /home/jc3/miniconda2/etc/profile.d/conda.sh
conda activate pytorch-1.1.0

module load gcc
module load cuda

sh run_r_101_d_8_deeplabv3_train_contrast_RMI.sh train 'deeplabv3-contrast-40k'
