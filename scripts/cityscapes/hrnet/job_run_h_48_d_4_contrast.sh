#!/bin/bash
#BSUB -n 16
#BSUB -W 24:00
#BSUB -R "rusage[mem=4000,ngpus_excl_p=4,scratch=10000]"
#BSUB -R "select[gpu_model0=GeForceRTX2080Ti]"
#BSUB -J "hrnet_contrast_dim512"
#BSUB -B
#BSUB -N
#BSUB -oo logs/

# activate env
source ../../../../pytorch-1.7.1/bin/activate

# copy data
rsync -aP /cluster/work/cvl/tiazhou/data/CityscapesZIP/openseg.tar ${TMPDIR}/
mkdir ${TMPDIR}/Cityscapes
tar -xf ${TMPDIR}/openseg.tar -C ${TMPDIR}/Cityscapes

# copy assets
rsync -aP /cluster/work/cvl/tiazhou/assets/openseg/hrnetv2_w48_imagenet_pretrained.pth ${TMPDIR}/hrnetv2_w48_imagenet_pretrained.pth

# define scratch dir
SCRATCH_DIR="/cluster/scratch/tiazhou/Openseg"

sh run_h_48_d_4_contrast.sh train 'hrnet_contrast_dim512' ${TMPDIR} ${SCRATCH_DIR}
