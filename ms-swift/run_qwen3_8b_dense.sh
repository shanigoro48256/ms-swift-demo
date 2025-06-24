#!/bin/bash
# run_qwen3_8b_dense.sh

RUN_ID="$(date +%Y%m%d-%H%M)"
MODEL_NAME="Qwen3-8B-Base-mcore"
DATA_PATH="data/apollo_cpt.jsonl"
TP_SIZE=2
GLOBAL_BS=512
MB_SIZE=2
LR=1e-5
WARMUP_ITERS=100
TRAIN_ITERS=1000
LR_MIN=1e-6
SAVE_INT=50
DS_NUMPROC=64
WANDB_PROJ="ms-swift"
LOG_INT=1

SAFE_MODEL_TAG="$(echo "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]' | tr -s ' _' '-')"
CKPT_DIR="checkpoints/${SAFE_MODEL_TAG}-${RUN_ID}"
LOG_DIR="logs/${SAFE_MODEL_TAG}-${RUN_ID}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

CUDA_DEVICE_MAX_CONNECTIONS=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
megatron pt \
  --load "${MODEL_NAME}" \
  --dataset "${DATA_PATH}" \
  --tensor_model_parallel_size "${TP_SIZE}" \
  --context_parallel_size 1 \
  --sequence_parallel true \
  --micro_batch_size "${MB_SIZE}" \
  --global_batch_size "${GLOBAL_BS}" \
  --recompute_granularity full \
  --recompute_method uniform \
  --recompute_num_layers 1 \
  --train_iters "${TRAIN_ITERS}" \
  --finetune true \
  --cross_entropy_loss_fusion true \
  --lr "${LR}" \
  --lr_warmup_iters "${WARMUP_ITERS}" \
  --min_lr "${LR_MIN}" \
  --save "${CKPT_DIR}" \
  --save_interval "${SAVE_INT}" \
  --max_length 32768 \
  --truncation_strategy right \
  --num_workers 1 \
  --no_save_optim true \
  --no_save_rng true \
  --dataset_num_proc "${DS_NUMPROC}" \
  --packing true \
  --streaming true \
  --use_flash_attn true \
  --wandb_project "${WANDB_PROJ}" \
  --wandb_exp_name "${SAFE_MODEL_TAG}-${RUN_ID}" \
  --log_interval "${LOG_INT}"

