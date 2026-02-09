#!/bin/bash
# Chla FNO-CBAM 目标区域训练启动脚本
# 8卡 DDP 分布式训练

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 配置
DATA_DIR="/data_new/chla_data_imputation_data_260125/chla_data_pretraining/filled_target_modified"
SST_DIR="/data_new/chla_data_imputation_data_260125/chla_data_pretraining/sst_daily_fusion_target_modified/"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/chla_target_train"
RESUME_CKPT=""
NUM_GPUS=8

# 训练参数
EPOCHS=100
BATCH_SIZE=16  # per GPU
LR=2e-4
MASK_RATIO=0.25

# 训练年份
TRAIN_YEARS="2016,2017,2018,2019,2020,2021,2022,2023"
VAL_YEARS="2024"

echo "============================================================"
echo "Chla FNO-CBAM Target Training"
echo "============================================================"
echo "Data dir: ${DATA_DIR}"
echo "Resume checkpoint: ${RESUME_CKPT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS}"
echo "Learning rate: ${LR}"
echo "============================================================"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agent

# 运行训练
cd "${PROJECT_ROOT}"

torchrun --nproc_per_node=${NUM_GPUS} \
    --master_port=29501 \
    training/train_chla.py \
    --data_dir ${DATA_DIR} \
    --sst_dir ${SST_DIR} \
    --output_dir ${OUTPUT_DIR} \
    ${RESUME_CKPT:+--resume ${RESUME_CKPT}} \
    --years_train ${TRAIN_YEARS} \
    --years_val ${VAL_YEARS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --mask_ratio ${MASK_RATIO} \
    --use_amp \
    --resample_by_missing \
    --resample_gamma 1.5 \
    --num_workers 8 \
    --log_interval 20 \
    --save_interval 10

echo "============================================================"
echo "Training completed!"
echo "============================================================"
