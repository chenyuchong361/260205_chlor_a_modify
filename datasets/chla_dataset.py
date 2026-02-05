#!/usr/bin/env python3
"""
目标区域训练数据集

Author: Claude
Date: 2025-01-30
"""

import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional
import random
import warnings
warnings.filterwarnings('ignore')


def generate_artificial_mask(
    missing_mask: np.ndarray,
    mask_ratio: float = 0.2,
    min_hole_size: int = 5,
    max_hole_size: int = 30,
) -> np.ndarray:
    """在观测区域生成人工挖空掩码"""
    H, W = missing_mask.shape
    observed_mask = (missing_mask == 0)
    num_observed = observed_mask.sum()

    if num_observed == 0:
        return np.zeros_like(missing_mask)

    target_masked = int(num_observed * mask_ratio)
    artificial_mask = np.zeros_like(missing_mask)
    current_masked = 0

    obs_rows, obs_cols = np.where(observed_mask)
    if len(obs_rows) == 0:
        return artificial_mask

    row_min, row_max = obs_rows.min(), obs_rows.max()
    col_min, col_max = obs_cols.min(), obs_cols.max()

    avg_hole_area = ((min_hole_size + max_hole_size) / 2) ** 2
    num_holes = max(1, int(target_masked / avg_hole_area * 1.5))
    max_attempts = num_holes * 10
    attempts = 0

    while current_masked < target_masked and attempts < max_attempts:
        attempts += 1
        hole_h = np.random.randint(min_hole_size, max_hole_size + 1)
        hole_w = np.random.randint(min_hole_size, max_hole_size + 1)

        if row_max - hole_h < row_min or col_max - hole_w < col_min:
            continue

        row_start = np.random.randint(row_min, max(row_min + 1, row_max - hole_h + 1))
        col_start = np.random.randint(col_min, max(col_min + 1, col_max - hole_w + 1))

        hole_region = observed_mask[row_start:row_start+hole_h, col_start:col_start+hole_w]
        if hole_region.sum() < hole_h * hole_w * 0.3:
            continue

        artificial_mask[row_start:row_start+hole_h, col_start:col_start+hole_w] = 1
        current_masked = (artificial_mask & observed_mask).sum()

    artificial_mask = artificial_mask & observed_mask.astype(np.int8)
    return artificial_mask


class ChlaFinetuneDataset(Dataset):
    """
    目标域微调数据集

    直接使用181×141目标域数据
    """

    # 共享mask_ratio文件
    _mask_ratio_file = "/tmp/chla_finetune_mask_ratio.txt"
    _default_mask_ratio = 0.2

    @classmethod
    def set_mask_ratio(cls, ratio: float):
        """设置共享mask_ratio"""
        try:
            with open(cls._mask_ratio_file, 'w') as f:
                f.write(str(ratio))
        except:
            pass

    @classmethod
    def get_mask_ratio(cls) -> float:
        """获取共享mask_ratio"""
        try:
            with open(cls._mask_ratio_file, 'r') as f:
                return float(f.read().strip())
        except:
            return cls._default_mask_ratio

    def __init__(
        self,
        data_dir: str,
        years: List[int] = None,
        window_size: int = 30,
        mask_ratio: float = 0.2,
        augment: bool = True,
        min_valid_ratio: float = 0.1,
        preload: bool = True,
    ):
        """
        Args:
            data_dir: 目标域数据目录
            years: 使用的年份列表
            window_size: 时间窗口
            mask_ratio: 挖空比例
            augment: 数据增强
            min_valid_ratio: 最小有效率
            preload: 预加载到内存
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self._mask_ratio = mask_ratio
        self.set_mask_ratio(mask_ratio)
        self.augment = augment
        self.min_valid_ratio = min_valid_ratio

        # 收集日期
        self.dates = self._collect_dates(years)
        print(f"Found {len(self.dates)} dates")

        # 构建样本列表
        self.samples = self.dates[self.window_size - 1:]
        print(f"Built {len(self.samples)} samples")

        # 预加载数据到内存
        self.preload = preload
        self.data_cache = {}
        if preload:
            self._preload_all_data()

    def _preload_all_data(self):
        """预加载所有数据到内存"""
        print("Preloading all data into memory...")
        loaded = 0
        for date_str in self.dates:
            h5_path = self._get_h5_path(date_str)
            try:
                with h5py.File(h5_path, 'r') as f:
                    self.data_cache[date_str] = (
                        f['daily_chla_norm'][:],
                        f['daily_filled'][:],
                        f['missing_mask'][:]
                    )
                loaded += 1
            except Exception as e:
                print(f"Warning: Failed to load {h5_path}: {e}")
            if loaded % 1000 == 0:
                print(f"  Loaded {loaded}/{len(self.dates)} files...")
        print(f"Preloaded {loaded} files into memory")

    @property
    def mask_ratio(self):
        return self.get_mask_ratio()

    @mask_ratio.setter
    def mask_ratio(self, value):
        self._mask_ratio = value
        self.set_mask_ratio(value)

    def _collect_dates(self, years: List[int]) -> List[str]:
        """收集可用日期"""
        dates = []
        for year_dir in sorted(self.data_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            year = int(year_dir.name)
            if years is not None and year not in years:
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                for h5_file in sorted(month_dir.glob("*.h5")):
                    dates.append(h5_file.stem)
        return dates

    def _get_h5_path(self, date_str: str) -> Path:
        """获取H5文件路径"""
        year = date_str[:4]
        month = date_str[4:6]
        return self.data_dir / year / month / f"{date_str}.h5"

    def _load_data(self, date_str: str):
        """加载数据"""
        if self.preload and date_str in self.data_cache:
            return self.data_cache[date_str]
        h5_path = str(self._get_h5_path(date_str))
        with h5py.File(h5_path, 'r') as f:
            return (
                f['daily_chla_norm'][:],
                f['daily_filled'][:],
                f['missing_mask'][:]
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        target_date = self.samples[idx]
        date_idx = self.dates.index(target_date)
        start_idx = date_idx - self.window_size + 1

        if start_idx < 0:
            return self.__getitem__((idx + 1) % len(self))

        date_sequence = self.dates[start_idx:date_idx + 1]

        # 加载30天数据
        seq_filled = []
        seq_mask = []

        for date_str in date_sequence:
            try:
                _, daily_filled, missing_mask = self._load_data(date_str)
                seq_filled.append(daily_filled)
                seq_mask.append(missing_mask)
            except Exception as e:
                return self.__getitem__((idx + 1) % len(self))

        seq_filled = np.stack(seq_filled, axis=0).astype(np.float32)
        seq_mask = np.stack(seq_mask, axis=0).astype(np.float32)

        # 目标日数据
        target_norm, target_filled, target_missing = self._load_data(target_date)

        # 检查有效率
        valid_ratio = 1 - target_missing.mean()
        if valid_ratio < self.min_valid_ratio:
            return self.__getitem__((idx + 1) % len(self))

        # 人工挖空
        artificial_mask = generate_artificial_mask(
            target_missing.astype(np.int8),
            mask_ratio=self.mask_ratio
        ).astype(np.float32)

        # 更新目标日
        input_mask_target = np.maximum(target_missing, artificial_mask)
        seq_mask[-1] = input_mask_target
        seq_filled[-1] = np.where(artificial_mask == 1, 0.5, seq_filled[-1])

        # 数据增强
        if self.augment:
            if np.random.rand() > 0.5:
                seq_filled = np.flip(seq_filled, axis=2).copy()
                seq_mask = np.flip(seq_mask, axis=2).copy()
                target_norm = np.flip(target_norm, axis=1).copy()
                artificial_mask = np.flip(artificial_mask, axis=1).copy()
                target_missing = np.flip(target_missing, axis=1).copy()

            if np.random.rand() > 0.5:
                seq_filled = np.flip(seq_filled, axis=1).copy()
                seq_mask = np.flip(seq_mask, axis=1).copy()
                target_norm = np.flip(target_norm, axis=0).copy()
                artificial_mask = np.flip(artificial_mask, axis=0).copy()
                target_missing = np.flip(target_missing, axis=0).copy()

        input_tensor = np.concatenate([seq_filled, seq_mask], axis=0)

        return {
            'input': torch.from_numpy(input_tensor),
            'target': torch.from_numpy(target_norm.astype(np.float32)),
            'artificial_mask': torch.from_numpy(artificial_mask),
            'missing_mask': torch.from_numpy(target_missing.astype(np.float32)),
            'date': target_date
        }


if __name__ == '__main__':
    print("Testing ChlaFinetuneDataset...")

    dataset = ChlaFinetuneDataset(
        data_dir='/data_new/chla_data_imputation_data_260125/chla_data_pretraining/filled_target_modified',
        years=[2020],
        window_size=30,
        mask_ratio=0.2,
        augment=False,
        preload=True
    )

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Input shape: {sample['input'].shape}")
    print(f"Target shape: {sample['target'].shape}")
    print(f"Date: {sample['date']}")
