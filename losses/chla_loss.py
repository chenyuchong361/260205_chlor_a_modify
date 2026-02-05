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

    def __init__(
        self,
        lambda_base: float = 0.0,
        use_mae: bool = True,
        min_supervised_pixels: int = 0,
        low_supervision_weight: float = 1.0,
    ):
        super().__init__()
        self.lambda_base = float(lambda_base)
        self.use_mae = bool(use_mae)
        self.min_supervised_pixels = int(min_supervised_pixels)
        self.low_supervision_weight = float(low_supervision_weight)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        artificial_mask: torch.Tensor,
        missing_mask: torch.Tensor,
        baseline: torch.Tensor = None,
    ) -> dict:
        # 兼容 [B, 1, H, W]
        if pred.dim() == 4:
            pred = pred.squeeze(1)

        eps = 1e-8

        # 仅在人工挖空区域计算 MAE
        recon_mask = artificial_mask.float()
        num_recon_pixels = recon_mask.sum()

        if num_recon_pixels > 0:
            recon_loss = (torch.abs(pred - target) * recon_mask).sum() / (num_recon_pixels + eps)
        else:
            recon_loss = torch.tensor(0.0, device=pred.device)

        # 保持训练脚本兼容：依然返回 base_loss，但不参与训练
        base_loss = torch.tensor(0.0, device=pred.device)
        total_loss = recon_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'base_loss': base_loss,
            'num_recon_pixels': num_recon_pixels,
        }



