#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agent

python3 "${PROJECT_ROOT}/visualization/plot_simple_4panel.py" \
    --date 20241001 \
    --data-dir "/data_new/chla_data_imputation_data_260125/chla_data_pretraining/filled_target_modified" \
    --sst-dir "/data_new/chla_data_imputation_data_260125/chla_data_pretraining/sst_daily_fusion_target_modified/" \
    --model "${PROJECT_ROOT}/checkpoints/chla_target_train/best_model.pth" \
    --output "${PROJECT_ROOT}/visualization/output/chla_plot/" \
    --mask-ratio 0.2 \
    --seed 42 \
    --gpu 0


