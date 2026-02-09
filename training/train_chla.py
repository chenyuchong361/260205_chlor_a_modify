#!/usr/bin/env python3
"""
叶绿素 FNO-CBAM 目标区域训练脚本

使用目标域全时段数据直接训练：
- 支持断点续训
- 高缺失天加权采样
- 较小学习率

Author: Claude
Date: 2025-01-30
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fno_cbam_chla import FNO_CBAM_Chla
from datasets.chla_dataset import ChlaFinetuneDataset
from losses.chla_loss import ChlaReconstructionLoss


class Logger:
    """双向日志记录器：同时输出到控制台和文件"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def setup_distributed():
    """初始化分布式训练"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_model(args):
    """创建模型"""
    model = FNO_CBAM_Chla(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        modes1=args.modes1,
        modes2=args.modes2,
        width=args.width,
        depth=args.depth,
    )
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, args, rank=0):
    """训练一个epoch"""
    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        inputs = batch['input'].to(device)
        target = batch['target'].to(device)
        artificial_mask = batch['artificial_mask'].to(device)

        optimizer.zero_grad()

        with autocast(enabled=args.use_amp):
            pred = model(inputs)
            loss_dict = criterion(
                pred=pred,
                target=target,
                artificial_mask=artificial_mask,
            )
            loss = loss_dict['loss']

        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if rank == 0 and batch_idx % args.log_interval == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: loss={loss.item():.4f}")

    return {
        'loss': total_loss / num_batches,
    }


def validate(model, dataloader, criterion, device, args):
    """验证"""
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            target = batch['target'].to(device)
            artificial_mask = batch['artificial_mask'].to(device)

            with autocast(enabled=args.use_amp):
                pred = model(inputs)
                loss_dict = criterion(
                    pred=pred,
                    target=target,
                    artificial_mask=artificial_mask,
                )

            total_loss += loss_dict['loss'].item()
            num_batches += 1

    return {
        'loss': total_loss / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Chla FNO-CBAM Target Training')

    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Target domain filled data directory')
    parser.add_argument('--sst_dir', type=str, required=True,
                       help='SST data directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/finetune',
                       help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume checkpoint path')
    parser.add_argument('--years_train', type=str, default='2016,2017,2018,2019,2020,2021,2022,2023',
                       help='Training years')
    parser.add_argument('--years_val', type=str, default='2024',
                       help='Validation years')

    # 模型参数
    parser.add_argument('--in_channels', type=int, default=152)
    parser.add_argument('--out_channels', type=int, default=1)
    parser.add_argument('--modes1', type=int, default=36)
    parser.add_argument('--modes2', type=int, default=28)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--depth', type=int, default=6)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true')

    # 损失参数
    parser.add_argument('--mask_ratio', type=float, default=0.25)
    parser.add_argument('--smooth_label', action='store_true',
                       help='Apply Gaussian smoothing to target labels before computing loss')
    parser.add_argument('--smooth_kernel_size', type=int, default=5,
                       help='Kernel size for Gaussian smoothing (must be odd)')
    parser.add_argument('--smooth_sigma', type=float, default=1.0,
                       help='Sigma for Gaussian smoothing')

    # 采样参数
    parser.add_argument('--resample_by_missing', action='store_true',
                       help='Resample by missing ratio')
    parser.add_argument('--resample_gamma', type=float, default=1.5)

    # 其他
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--save_interval', type=int, default=10)

    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    # 创建输出目录和日志
    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(args.output_dir) / f"train_log_{timestamp}.txt"
        logger = Logger(log_file)
        sys.stdout = logger

        print("=" * 60)
        print("Chla FNO-CBAM Target Training")
        print("=" * 60)
        print(f"Timestamp: {timestamp}")
        print(f"Log file: {log_file}")
        print(f"World size: {world_size}")
        print(f"Data dir: {args.data_dir}")
        print(f"SST dir: {args.sst_dir}")
        print(f"Resume checkpoint: {args.resume}")
        print(f"Output dir: {args.output_dir}")
        print(f"Learning rate: {args.lr}")
        print(f"Mask ratio: {args.mask_ratio}")
        print(f"Smooth label: {args.smooth_label}")
        if args.smooth_label:
            print(f"  Smooth kernel size: {args.smooth_kernel_size}")
            print(f"  Smooth sigma: {args.smooth_sigma}")
        print("=" * 60)

    years_train = [int(y) for y in args.years_train.split(',')]
    years_val = [int(y) for y in args.years_val.split(',')]

    # 数据集
    train_dataset = ChlaFinetuneDataset(
        data_dir=args.data_dir,
        sst_dir=args.sst_dir,
        years=years_train,
        window_size=30,
        mask_ratio=args.mask_ratio,
        augment=True,
        preload=True,
    )

    val_dataset = ChlaFinetuneDataset(
        data_dir=args.data_dir,
        sst_dir=args.sst_dir,
        years=years_val,
        window_size=30,
        mask_ratio=args.mask_ratio,
        augment=False,
        preload=True,
    )

    if rank == 0:
        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Val dataset: {len(val_dataset)} samples")

    # 采样器设置
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 模型
    model = get_model(args).to(device)
    resume_checkpoint = None

    if args.resume:
        if rank == 0:
            print(f"Resuming from {args.resume}")
        resume_checkpoint = torch.load(args.resume, map_location=device)
        if 'model_state_dict' in resume_checkpoint:
            state_dict = resume_checkpoint['model_state_dict']
        elif 'model' in resume_checkpoint:
            state_dict = resume_checkpoint['model']
        else:
            state_dict = resume_checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    criterion = ChlaReconstructionLoss(
        smooth_label=args.smooth_label,
        smooth_kernel_size=args.smooth_kernel_size,
        smooth_sigma=args.smooth_sigma
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler() if args.use_amp else None

    best_val_loss = float('inf')
    start_epoch = 0

    if resume_checkpoint:
        if 'optimizer' in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint['optimizer'])
        if 'scheduler' in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint['scheduler'])
        start_epoch = resume_checkpoint.get('epoch', -1) + 1
        best_val_loss = resume_checkpoint.get('best_val_loss', best_val_loss)

    for epoch in range(start_epoch, args.epochs):
        # 注意：只有DistributedSampler有set_epoch方法，WeightedRandomSampler没有
        if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, args, rank
        )

        val_metrics = validate(model, val_loader, criterion, device, args)
        scheduler.step()

        if rank == 0:
            print(f"  Train - loss: {train_metrics['loss']:.4f}")
            print(f"  Val   - loss: {val_metrics['loss']:.4f}")

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_path = Path(args.output_dir) / 'best_model.pth'
                state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model': state_dict,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'args': vars(args),
                }, save_path)
                print(f"  Saved best model to {save_path}")

            if (epoch + 1) % args.save_interval == 0:
                save_path = Path(args.output_dir) / f'checkpoint_epoch{epoch + 1}.pth'
                state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model': state_dict,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'args': vars(args),
                }, save_path)

    cleanup_distributed()

    if rank == 0:
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("=" * 60)

        # 关闭日志
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
            sys.stdout = sys.__stdout__


if __name__ == '__main__':
    main()
