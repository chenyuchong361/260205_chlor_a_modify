#!/bin/bash
#======================================================================
# 查看最新日志的便捷脚本
#======================================================================

echo "========================================"
echo "Latest Logs Quick Access"
echo "========================================"

# 查找最新的训练日志
echo ""
echo "📝 Latest Training Logs:"
find checkpoints/ -name "train_log_*.txt" -type f 2>/dev/null | sort -r | head -3 | while read log; do
    echo "  - $log"
    echo "    Size: $(du -h "$log" | cut -f1)"
    echo "    Modified: $(stat -c %y "$log" | cut -d' ' -f1,2 | cut -d'.' -f1)"
done

# 查找最新的推理日志
echo ""
echo "🔍 Latest Inference Logs:"
find outputs/ -name "inference_log_*.txt" -type f 2>/dev/null | sort -r | head -3 | while read log; do
    echo "  - $log"
    echo "    Size: $(du -h "$log" | cut -f1)"
    echo "    Modified: $(stat -c %y "$log" | cut -d'.' -f1)"
done

echo ""
echo "========================================"
echo "Usage:"
echo "  cat <log_file>           # View log"
echo "  tail -f <log_file>       # Monitor in real-time"
echo "  less <log_file>          # Browse log"
echo "========================================"
