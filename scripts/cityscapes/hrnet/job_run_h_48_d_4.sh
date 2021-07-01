#!/bin/bash
#BSUB -n 16
#BSUB -W 48:00
#BSUB -R "rusage[mem=4000,ngpus_excl_p=4,scratch=10000]"
#BSUB -R "select[gpu_model0=GeForceRTX2080Ti]"
#BSUB -J "hrnet_ce_80k"
#BSUB -B
#BSUB -N
#BSUB -oo logs/

# activate env
source ../../../pytorch-1.7.1/bin/activate

# copy data
rsync -aP /cluster/work/cvl/tiazhou/data/CityscapesZIP/openseg.tar ${TMPDIR}/
mkdir ${TMPDIR}/Cityscapes
tar -xf ${TMPDIR}/openseg.tar -C ${TMPDIR}/Cityscapes

# copy assets
rsync -aP /cluster/work/cvl/tiazhou/assets/openseg/hrnetv2_w48_imagenet_pretrained.pth ${TMPDIR}/hrnetv2_w48_imagenet_pretrained.pth

# define scratch dir
SCRATCH_DIR="/cluster/scratch/tiazhou/Openseg"

sh run_h_48_d_4.sh train 'hrnet_ce_80k' ${TMPDIR} ${SCRATCH_DIR}
