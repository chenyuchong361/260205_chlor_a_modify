#!/usr/bin/env python3
"""\
叶绿素缺失值重建损失函数（简化版：Masked MAE）

训练时仅在人工挖空区域（artificial_mask==1）计算 MAE。

Author: Claude
Date: 2025-01-30
"""

import torch
import torch.nn as nn


class ChlaReconstructionLoss(nn.Module):
    """叶绿素重建损失（Masked MAE）"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        artificial_mask: torch.Tensor,
    ) -> dict:
        # 兼容 [B, 1, H, W]
        if pred.dim() == 4:
            pred = pred.squeeze(1)

        eps = 1e-8

        # 仅在人工挖空区域计算 MAE
        recon_mask = artificial_mask.float()
        num_recon_pixels = recon_mask.sum()

        if num_recon_pixels > 0:
            loss = (torch.abs(pred - target) * recon_mask).sum() / (num_recon_pixels + eps)
        else:
            loss = torch.tensor(0.0, device=pred.device)

        return {
            'loss': loss,
            'num_recon_pixels': num_recon_pixels,
        }



