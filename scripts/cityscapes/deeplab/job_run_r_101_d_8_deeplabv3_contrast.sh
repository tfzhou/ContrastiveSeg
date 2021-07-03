#!/bin/bash
#BSUB -n 16
#BSUB -W 24:00
#BSUB -R "rusage[mem=4000,ngpus_excl_p=4,scratch=10000]"
#BSUB -R "select[gpu_model0=GeForceRTX2080Ti]"
#BSUB -J "deeplab_v3_contrast"
#BSUB -B
#BSUB -N
#BSUB -oo logs/


source ../../../../pytorch-1.7.1/bin/activate

# copy data
rsync -aP /cluster/work/cvl/tiazhou/data/CityscapesZIP/openseg.tar ${TMPDIR}/
mkdir ${TMPDIR}/Cityscapes/
tar -xf ${TMPDIR}/openseg.tar -C ${TMPDIR}/Cityscapes

# copy assets
rsync -aP /cluster/work/cvl/tiazhou/assets/openseg/resnet101-imagenet.pth ${TMPDIR}/resnet101-imagenet.pth

# define scratch dir
SCRATCH_DIR="/cluster/scratch/tiazhou/Openseg"

sh run_r_101_d_8_deeplabv3_contrast_train.sh train 'deeplab_v3_contrast' ${TMPDIR} ${SCRATCH_DIR} 'ss'
