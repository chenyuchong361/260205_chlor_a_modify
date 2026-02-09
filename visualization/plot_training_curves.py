#!/usr/bin/env python3
"""
Training Curves Visualization
绘制训练过程中的loss曲线和学习率变化
Author: Claude
Date: 2026-02-06
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def parse_training_log(log_file):
    """
    解析训练日志文件

    Args:
        log_file: 训练日志文件路径

    Returns:
        dict: 包含训练数据的字典
    """
    epochs = []
    train_losses = []
    val_losses = []
    learning_rates = []

    # 用于存储batch级别的loss（可选）
    batch_data = []

    with open(log_file, 'r', encoding='utf-8') as f:
        current_epoch = None
        current_lr = None

        for line in f:
            line = line.strip()

            # 匹配 Epoch 行: "Epoch 1/10"
            epoch_match = re.match(r'Epoch (\d+)/(\d+)', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            # 匹配学习率行: "  LR: 0.000200"
            lr_match = re.match(r'\s*LR:\s+([\d.e-]+)', line)
            if lr_match:
                current_lr = float(lr_match.group(1))
                continue

            # 匹配训练loss: "  Train - loss: 1.2664, recon: 1.2664"
            train_match = re.match(r'\s*Train - loss:\s+([\d.]+)', line)
            if train_match and current_epoch is not None:
                train_loss = float(train_match.group(1))
                train_losses.append(train_loss)
                continue

            # 匹配验证loss: "  Val   - loss: 0.3548, recon: 0.3548"
            val_match = re.match(r'\s*Val\s*-\s*loss:\s+([\d.]+)', line)
            if val_match and current_epoch is not None:
                val_loss = float(val_match.group(1))
                val_losses.append(val_loss)
                epochs.append(current_epoch)
                if current_lr is not None:
                    learning_rates.append(current_lr)
                continue

            # 匹配batch loss: "  Batch 0/45: loss=4.7017, recon=4.7017"
            batch_match = re.match(r'\s*Batch (\d+)/\d+:\s*loss=([\d.]+)', line)
            if batch_match and current_epoch is not None:
                batch_idx = int(batch_match.group(1))
                batch_loss = float(batch_match.group(2))
                batch_data.append({
                    'epoch': current_epoch,
                    'batch': batch_idx,
                    'loss': batch_loss
                })

    return {
        'epochs': np.array(epochs),
        'train_losses': np.array(train_losses),
        'val_losses': np.array(val_losses),
        'learning_rates': np.array(learning_rates),
        'batch_data': batch_data
    }


def plot_training_curves(data, output_path, title='Training Curves', show_batch=False):
    """
    绘制训练曲线
    - 验证集曲线改为实心圆点
    - 纵坐标保持对数刻度

    Args:
        data: parse_training_log返回的数据字典
        output_path: 输出图片路径
        title: 图片标题
        show_batch: 是否显示batch级别的loss
    """
    epochs = data['epochs']
    train_losses = data['train_losses']
    val_losses = data['val_losses']
    learning_rates = data['learning_rates']

    # 创建图形
    if show_batch and len(data['batch_data']) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        axes = [axes[0], axes[1]]

    # 1. 训练和验证Loss曲线
    ax1 = axes[0] if not show_batch else axes[0, 0]
    
    # 训练集保持空心圆点 (便于区分)
    ax1.plot(epochs, train_losses, 'o-', label='Train Loss',
             color='#2E86AB', linewidth=2, markersize=6, markerfacecolor='white', markeredgewidth=2)
    
    # --- 修改点：验证集改为实心圆点 ---
    # 'o-' 表示圆形标记，移除了 markerfacecolor='white' 使其变为实心
    ax1.plot(epochs, val_losses, 'o-', label='Val Loss',
             color='#A23B72', linewidth=2, markersize=6)
    # -------------------------------

    # 标记最优验证loss
    best_val_idx = np.argmin(val_losses)
    best_val_epoch = epochs[best_val_idx]
    best_val_loss = val_losses[best_val_idx]
    ax1.scatter([best_val_epoch], [best_val_loss], color='red', s=150,
                marker='*', zorder=5, label=f'Best Val ({best_val_loss:.4f})')

    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss (Log Scale)', fontsize=12, fontweight='bold')
    ax1.set_title('Loss Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)
    ax1.set_yscale('log') # 保持对数坐标

    # 2. 学习率曲线
    ax2 = axes[1] if not show_batch else axes[0, 1]
    if len(learning_rates) > 0:
        ax2.plot(epochs, learning_rates, 'o-',
                color='#F18F01', linewidth=2, markersize=6, markerfacecolor='white', markeredgewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(left=0)
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    else:
        ax2.text(0.5, 0.5, 'No Learning Rate Data',
                ha='center', va='center', fontsize=14, color='gray')
        ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')

    # 3. Batch级别loss (可选)
    if show_batch and len(data['batch_data']) > 0:
        ax3 = axes[1, 0]
        batch_data = data['batch_data']

        # 按epoch分组绘制
        unique_epochs = sorted(set(d['epoch'] for d in batch_data))
        for ep in unique_epochs:
            ep_data = [d for d in batch_data if d['epoch'] == ep]
            batches = [d['batch'] for d in ep_data]
            losses = [d['loss'] for d in ep_data]

            # 计算全局batch索引（用于x轴）
            max_batch = max(batches) + 1
            global_batches = [(ep - 1) * max_batch + b for b in batches]

            ax3.plot(global_batches, losses, 'o-', alpha=0.6, markersize=3, linewidth=1)

        ax3.set_xlabel('Batch (Global)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax3.set_title('Batch-level Loss (All Epochs)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xlim(left=0)
        ax3.set_ylim(bottom=0)

        # 4. Loss分布箱线图
        ax4 = axes[1, 1]
        epoch_losses_list = []
        epoch_labels = []
        for ep in unique_epochs:
            ep_losses = [d['loss'] for d in batch_data if d['epoch'] == ep]
            epoch_losses_list.append(ep_losses)
            epoch_labels.append(f'E{ep}')

        bp = ax4.boxplot(epoch_losses_list, labels=epoch_labels,
                        patch_artist=True, showmeans=True, meanline=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#A8DADC')
            patch.set_alpha(0.7)

        ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax4.set_title('Loss Distribution per Epoch', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax4.set_ylim(bottom=0)

        # 旋转x轴标签
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved to: {output_path}")
    plt.close()


def print_summary(data):
    """打印训练摘要统计"""
    epochs = data['epochs']
    train_losses = data['train_losses']
    val_losses = data['val_losses']

    if len(epochs) == 0:
        print("⚠ No training data found in log file")
        return

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Total Epochs: {len(epochs)}")
    print(f"\nTrain Loss:")
    print(f"  Initial: {train_losses[0]:.4f}")
    print(f"  Final:   {train_losses[-1]:.4f}")
    print(f"  Min:     {train_losses.min():.4f} (Epoch {epochs[np.argmin(train_losses)]})")
    print(f"  Mean:    {train_losses.mean():.4f}")

    print(f"\nValidation Loss:")
    print(f"  Initial: {val_losses[0]:.4f}")
    print(f"  Final:   {val_losses[-1]:.4f}")
    print(f"  Min:     {val_losses.min():.4f} (Epoch {epochs[np.argmin(val_losses)]})")
    print(f"  Mean:    {val_losses.mean():.4f}")

    # 计算改进率
    train_improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
    val_improvement = (val_losses[0] - val_losses[-1]) / val_losses[0] * 100

    print(f"\nImprovement:")
    print(f"  Train: {train_improvement:.2f}%")
    print(f"  Val:   {val_improvement:.2f}%")

    # 检查过拟合
    final_gap = abs(train_losses[-1] - val_losses[-1])
    print(f"\nFinal Train-Val Gap: {final_gap:.4f}")
    if val_losses[-1] > train_losses[-1] * 1.5:
        print("  ⚠ Potential overfitting detected")
    elif val_losses[-1] < train_losses[-1]:
        print("  ⚠ Unusual: Val loss < Train loss")
    else:
        print("  ✓ Reasonable gap")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plot training curves from log file')
    parser.add_argument('--log_file', type=str, required=True,
                       help='Path to training log file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output image path (default: log_file_dir/training_curves.png)')
    parser.add_argument('--title', type=str, default='Training Curves',
                       help='Plot title')
    parser.add_argument('--show_batch', action='store_true',
                       help='Show batch-level loss curves')
    parser.add_argument('--summary', action='store_true',
                       help='Print training summary statistics')

    args = parser.parse_args()

    log_file = Path(args.log_file)
    if not log_file.exists():
        print(f"✗ Error: Log file not found: {log_file}")
        return

    # 默认输出路径
    if args.output is None:
        output_dir = log_file.parent if log_file.parent != Path('.') else Path('.')
        output_path = output_dir / f"{log_file.stem}_training_curves.png"
    else:
        output_path = Path(args.output)

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Parsing training log: {log_file}")
    data = parse_training_log(log_file)

    if len(data['epochs']) == 0:
        print("✗ No training data found in log file")
        return

    print(f"Found {len(data['epochs'])} epochs of training data")
    if len(data['batch_data']) > 0:
        print(f"Found {len(data['batch_data'])} batch-level records")

    # 打印摘要
    if args.summary:
        print_summary(data)

    # 绘制曲线
    print(f"Plotting training curves...")
    plot_training_curves(data, output_path, title=args.title, show_batch=args.show_batch)

    print(f"\n✓ Done!")


if __name__ == '__main__':
    main()
