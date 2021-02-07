#!/bin/bash
#SBATCH -n 40
#SBATCH --output=log/%j-resnet101-ocr-contrast-40k-t0.07.out
#SBATCH --gres=gpu:v100:4
#SBATCH --time=48:00:00

source /home/jc3/miniconda2/etc/profile.d/conda.sh
conda activate pytorch-0.4.1

module load gcc
module load cuda

sh run_r_101_d_8_ocrnet_train_contrast.sh train 'resnet101-ocr-contrast-40k'
