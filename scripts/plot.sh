#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python3 "${PROJECT_ROOT}/visualization/plot_chla_4panel.py" \
    --region 1 \
    --date 20241001 \
    --mask-ratio 0.2 \
    --seed 42 \
    --output "${PROJECT_ROOT}/visualization/output/chla_plot/" \
    --model "${PROJECT_ROOT}/checkpoints/chla_target_train/best_model.pth" \
    --cropped-data-dir /data_new/chla_data_imputation_data_260125/chla_data_pretraining/cropped_regions/ \
    --gpu 0
