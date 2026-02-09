#!/usr/bin/env python3
"""\
叶绿素缺失值重建损失函数（简化版：Masked MAE）

训练时仅在人工挖空区域（artificial_mask==1）计算 MAE。

Author: Claude
Date: 2025-01-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def gaussian_kernel_2d(kernel_size: int, sigma: float) -> torch.Tensor:
    """生成2D高斯卷积核"""
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel_1d = gauss / gauss.sum()

    # 生成2D核
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d / kernel_2d.sum()

    return kernel_2d


class ChlaReconstructionLoss(nn.Module):
    """叶绿素重建损失（Masked MAE）"""

    def __init__(self, smooth_label: bool = True, smooth_kernel_size: int = 5, smooth_sigma: float = 1.0):
        """
        Args:
            smooth_label: 是否对label进行平滑
            smooth_kernel_size: 高斯核大小（必须是奇数）
            smooth_sigma: 高斯核标准差
        """
        super().__init__()
        self.smooth_label = smooth_label

        if smooth_label:
            # 预先创建高斯核
            kernel = gaussian_kernel_2d(smooth_kernel_size, smooth_sigma)
            # [1, 1, K, K] 格式用于conv2d
            self.register_buffer('gaussian_kernel', kernel.view(1, 1, smooth_kernel_size, smooth_kernel_size))
            self.padding = smooth_kernel_size // 2

    def _smooth_target(self, target: torch.Tensor) -> torch.Tensor:
        """对target进行高斯平滑"""
        # target: [B, H, W] 或 [H, W]
        if target.dim() == 2:
            target = target.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            squeeze_output = True
        elif target.dim() == 3:
            target = target.unsqueeze(1)  # [B, 1, H, W]
            squeeze_output = False
        else:
            squeeze_output = False

        # 确保gaussian_kernel在正确的设备上
        kernel = self.gaussian_kernel.to(target.device)

        # 应用高斯滤波
        smoothed = F.conv2d(
            target,
            kernel,
            padding=self.padding,
            groups=1
        )

        if squeeze_output:
            smoothed = smoothed.squeeze(0).squeeze(0)
        else:
            smoothed = smoothed.squeeze(1)

        return smoothed

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        artificial_mask: torch.Tensor,
    ) -> dict:
        # 兼容 [B, 1, H, W]
        if pred.dim() == 4:
            pred = pred.squeeze(1)

        # 对target进行平滑（如果启用）
        if self.smooth_label:
            target_smoothed = self._smooth_target(target)
        else:
            target_smoothed = target

        eps = 1e-8

        # 仅在人工挖空区域计算 MAE
        recon_mask = artificial_mask.float()
        num_recon_pixels = recon_mask.sum()

        if num_recon_pixels > 0:
            loss = (torch.abs(pred - target_smoothed) * recon_mask).sum() / (num_recon_pixels + eps)
        else:
            loss = torch.tensor(0.0, device=pred.device)

        return {
            'loss': loss,
            'num_recon_pixels': num_recon_pixels,
        }



