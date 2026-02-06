#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agent

python3 "${PROJECT_ROOT}/inference/infer_chla.py" \
    --data_dir "/data_new/chla_data_imputation_data_260125/chla_data_pretraining/filled_target_modified" \
    --sst_dir "/data_new/chla_data_imputation_data_260125/chla_data_pretraining/sst_daily_fusion_target" \
    --checkpoint "${PROJECT_ROOT}/checkpoints/chla_target_train/best_model.pth" \
    --output_dir "${PROJECT_ROOT}/outputs/chla_inference_2024" \
    --year_start 2024 \
    --year_end 2024
