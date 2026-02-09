#!/bin/bash
#======================================================================
# Training Curves Plotting Script
# 绘制训练过程中的loss曲线和学习率变化
#======================================================================

# 激活conda环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate agent

# 设置路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PLOT_SCRIPT="${PROJECT_ROOT}/visualization/plot_training_curves.py"

# 默认参数
LOG_FILE="/home/cyc/260205_chlor_a_modify/checkpoints/chla_target_train/train_log_20260208_012158.txt"
OUTPUT="${2:-}"
SHOW_BATCH="${3:-false}"

# 构建命令
CMD="python ${PLOT_SCRIPT} --log_file ${LOG_FILE} --summary"

if [ -n "$OUTPUT" ]; then
    CMD="${CMD} --output ${OUTPUT}"
fi

if [ "$SHOW_BATCH" = "true" ]; then
    CMD="${CMD} --show_batch"
fi

# 执行
echo "========================================"
echo "Training Curves Plotting"
echo "========================================"
echo "Log file: ${LOG_FILE}"
echo "Script: ${PLOT_SCRIPT}"
echo ""

eval $CMD

echo ""
echo "========================================"
echo "Done!"
echo "========================================"
