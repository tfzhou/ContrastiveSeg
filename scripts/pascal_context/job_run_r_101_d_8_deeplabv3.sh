#!/bin/bash
#BSUB -n 16
#BSUB -W 72:00
#BSUB -R "rusage[mem=4000,ngpus_excl_p=4,scratch=10000]"
#BSUB -R "select[gpu_model0=TITANRTX]"
#BSUB -J "deeplab_v3"
#BSUB -B
#BSUB -N
#BSUB -oo logs/

# activate env
#source /cluster/home/tiazhou/miniconda3/etc/profile.d/conda.sh
#conda activate pytorch-1.7.1

source ../../../pytorch-1.7.1/bin/activate

# copy data
rsync -aP /cluster/work/cvl/tiazhou/data/pascalcontext.tar ${TMPDIR}/
tar -xf ${TMPDIR}/pascalcontext.tar -C ${TMPDIR}/

# copy assets
rsync -aP /cluster/work/cvl/tiazhou/assets/openseg/resnet101-imagenet.pth ${TMPDIR}/resnet101-imagenet.pth

# define scratch dir
SCRATCH_DIR="/cluster/scratch/tiazhou/Openseg"

sh run_r_101_d_8_deeplabv3_train.sh train 'deeplab_v3' ${TMPDIR} ${SCRATCH_DIR} 'ss'
