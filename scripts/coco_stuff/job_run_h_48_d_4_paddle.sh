#!/bin/bash
#BSUB -n 16
#BSUB -W 72:00
#BSUB -R "rusage[mem=4000,ngpus_excl_p=4,scratch=10000]"
#BSUB -R "select[gpu_model0=TITANRTX]"
#BSUB -J "hrnet_contrast_paddle"
#BSUB -B
#BSUB -N
#BSUB -oo logs/

## activate env
#source /cluster/home/tiazhou/miniconda3/etc/profile.d/conda.sh
#conda activate pytorch-1.7.1

source ../../pytorch-1.7.1/bin/activate

# copy data
rsync -aP /cluster/work/cvl/tiazhou/data/cocostuff.tar ${TMPDIR}/
tar -xf ${TMPDIR}/cocostuff.tar -C ${TMPDIR}/

# copy assets
rsync -aP /cluster/work/cvl/tiazhou/assets/openseg/HRNet_W48_C_ssld_pretrained.pth ${TMPDIR}/HRNet_W48_C_ssld_pretrained.pth

# define scratch dir
SCRATCH_DIR="/cluster/scratch/tiazhou/Openseg"

sh run_h_48_d_4_paddle.sh val 'hrnet_paddle' ${TMPDIR} ${SCRATCH_DIR}
