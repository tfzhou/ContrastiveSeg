#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../

DATA_ROOT=$3
SCRATCH_ROOT=$4
ASSET_ROOT=${DATA_ROOT}

DATA_DIR="${DATA_ROOT}/pascalcontext"
SAVE_DIR="${SCRATCH_ROOT}/seg_results/pascal_context/"
BACKBONE="hrnet48"

CONFIGS="configs/pascal_context/H_48_D_4.json"
CONFIGS_TEST="configs/pascal_context/H_48_D_4_TEST.json"

MODEL_NAME="hrnet_w48"
LOSS_TYPE="fs_ce_loss"
CHECKPOINTS_ROOT="${SCRATCH_ROOT}/pascal_context/"
CHECKPOINTS_NAME="${MODEL_NAME}_${BACKBONE}_"$2
LOG_FILE="${SCRATCH_ROOT}/logs/pascal_context/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`

PRETRAINED_MODEL="${ASSET_ROOT}/HRNet_W48_C_ssld_pretrained.pth"
MAX_ITERS=60000


if [ "$1"x == "train"x ]; then
  python -u main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --nbb_mult 10 \
                       --phase train \
                       --gathered n \
                       --loss_balance y \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --gpu 0 1 2 3 \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --max_iters ${MAX_ITERS} \
                       --checkpoints_root ${CHECKPOINTS_ROOT} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --distributed \
                       --pretrained ${PRETRAINED_MODEL} \
                       --test_interval 10000 \
                       2>&1 | tee ${LOG_FILE}
                       

elif [ "$1"x == "resume"x ]; then
  python -u main_contrastive.py --configs ${CONFIGS} \
                       --drop_last y \
                       --nbb_mult 10 \
                       --phase train \
                       --gathered n \
                       --loss_balance y \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --max_iters ${MAX_ITERS} \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --gpu 0 1 2 3 \
                       --resume_continue y \
                       --checkpoints_root ${CHECKPOINTS_ROOT} \
                       --resume ./checkpoints/pascal_context/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                        2>&1 | tee -a ${LOG_FILE}


elif [ "$1"x == "val"x ]; then
  python -u main.py --configs ${CONFIGS_TEST} \
                       --data_dir ${DATA_DIR} \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test \
                       --gpu 0 1 2 3 \
                       --resume ${CHECKPOINTS_ROOT}/checkpoints/pascal_context/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/val/image \
                       --log_to_file n \
                       --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms

#  cd lib/metrics
#  ${PYTHON} -u ade20k_evaluator.py --configs ../../${CONFIGS_TEST} \
#                                   --pred_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms/label \
#                                   --gt_dir ${DATA_DIR}/val/label


elif [ "$1"x == "test"x ]; then
  if [ "$3"x == "ss"x ]; then
    echo "[single scale] test"
    ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test --gpu 0 1 2 3 --resume ./checkpoints/pascal_context/${CHECKPOINTS_NAME}_latest.pth \
                         --test_dir ${DATA_DIR}/test --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ss
  else
    echo "[multiple scale + flip] test"
    ${PYTHON} -u main.py --configs ${CONFIGS_TEST} --drop_last y \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test --gpu 0 1 2 3 --resume ./checkpoints/pascal_context/${CHECKPOINTS_NAME}_latest.pth \
                         --test_dir ${DATA_DIR}/test --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ms
  fi


else
  echo "$1"x" is invalid..."
fi
