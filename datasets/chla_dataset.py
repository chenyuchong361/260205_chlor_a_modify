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
import math
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
        sst_dir: str,
        years: List[int] = None,
        window_size: int = 30,
        mask_ratio: float = 0.2,
        augment: bool = True,
        min_valid_ratio: float = 0.1,
        preload: bool = True,
    ):
        """
        Args:
            data_dir: 目标域数据目录 (Chl-a)
            sst_dir: SST数据目录
            years: 使用的年份列表
            window_size: 时间窗口
            mask_ratio: 挖空比例
            augment: 数据增强
            min_valid_ratio: 最小有效率
            preload: 预加载到内存
        """
        self.data_dir = Path(data_dir)
        self.sst_dir = Path(sst_dir)
        self.window_size = window_size
        self._mask_ratio = mask_ratio
        self.set_mask_ratio(mask_ratio)
        self.augment = augment
        self.min_valid_ratio = min_valid_ratio
        
        # 预计算 Lat/Lon Grid (H=181, W=141)
        # 15.0N ~ 24.0N, 111.0E ~ 118.0E
        lat = np.linspace(15.0, 24.0, 181)
        lon = np.linspace(111.0, 118.0, 141)
        self.lat_grid, self.lon_grid = np.meshgrid(lat, lon, indexing='ij')
        
        # 归一化到 [-1, 1]
        self.lat_norm = (self.lat_grid - 19.5) / 4.5  # (x - mid) / half_range
        self.lon_norm = (self.lon_grid - 114.5) / 3.5
        
        # 转为 float32
        self.lat_norm = self.lat_norm.astype(np.float32)
        self.lon_norm = self.lon_norm.astype(np.float32)

        # 收集日期
        self.dates = self._collect_dates(years)
        print(f"Found {len(self.dates)} dates with valid Chl-a and SST")

        # 构建样本列表（验证30天窗口完整性）
        self.samples = self._build_valid_samples()
        print(f"Built {len(self.samples)} valid samples with complete 30-day windows")

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
            sst_path = self._get_sst_path(date_str)
            try:
                # Load Chl-a
                with h5py.File(h5_path, 'r') as f:
                    chla_norm = f['daily_chla_norm'][:]
                    daily_filled = f['daily_filled'][:]
                    missing_mask = f['missing_mask'][:]

                # Load SST
                with h5py.File(sst_path, 'r') as f:
                    # 尝试猜测 key
                    keys = list(f.keys())
                    sst_key = 'daily_sst' if 'daily_sst' in keys else keys[0]
                    sst_data = f[sst_key][:]
                    # 填充SST的NaN值
                    if np.isnan(sst_data).any():
                        ocean_mean = np.nanmean(sst_data)
                        if np.isnan(ocean_mean):
                            ocean_mean = 0.0
                        sst_data = np.nan_to_num(sst_data, nan=ocean_mean)

                self.data_cache[date_str] = (chla_norm, daily_filled, missing_mask, sst_data)
                loaded += 1
            except Exception as e:
                print(f"Warning: Failed to load {date_str}: {e}")
            if loaded % 1000 == 0 and loaded > 0:
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
        """收集可用日期（同时检查Chl-a和SST是否都存在）"""
        dates = []
        skipped_count = 0
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
                    date_str = h5_file.stem
                    # 检查对应的SST文件是否存在
                    sst_path = self._get_sst_path(date_str)
                    if sst_path.exists():
                        dates.append(date_str)
                    else:
                        skipped_count += 1
        if skipped_count > 0:
            print(f"Skipped {skipped_count} dates due to missing SST files")
        return dates

    def _build_valid_samples(self) -> List[str]:
        """构建样本列表，确保每个样本的30天窗口都有完整的SST数据"""
        valid_samples = []
        date_set = set(self.dates)  # 用于快速查找

        for i in range(self.window_size - 1, len(self.dates)):
            target_date = self.dates[i]
            start_idx = i - self.window_size + 1

            # 检查窗口内所有日期是否都存在
            window_dates = self.dates[start_idx:i+1]
            if len(window_dates) == self.window_size:
                # 所有日期都在date_set中（已经验证过SST存在）
                valid_samples.append(target_date)

        return valid_samples

    def _get_h5_path(self, date_str: str) -> Path:
        """获取H5文件路径"""
        year = date_str[:4]
        month = date_str[4:6]
        return self.data_dir / year / month / f"{date_str}.h5"

    def _get_sst_path(self, date_str: str) -> Path:
        """获取SST文件路径"""
        year = date_str[:4]
        month = date_str[4:6]
        return self.sst_dir / year / month / f"{date_str}.h5"

    def _load_data(self, date_str: str):
        """加载数据"""
        if self.preload and date_str in self.data_cache:
            return self.data_cache[date_str]

        h5_path = str(self._get_h5_path(date_str))
        sst_path = str(self._get_sst_path(date_str))

        with h5py.File(h5_path, 'r') as f:
            chla_norm = f['daily_chla_norm'][:]
            daily_filled = f['daily_filled'][:]
            missing_mask = f['missing_mask'][:]

        with h5py.File(sst_path, 'r') as f:
            keys = list(f.keys())
            sst_key = 'daily_sst' if 'daily_sst' in keys else keys[0]
            sst = f[sst_key][:]
            # 填充SST的NaN值（陆地等区域）
            # 使用海洋区域的均值填充，如果全是NaN则用0
            if np.isnan(sst).any():
                ocean_mean = np.nanmean(sst)
                if np.isnan(ocean_mean):
                    ocean_mean = 0.0
                sst = np.nan_to_num(sst, nan=ocean_mean)

        return chla_norm, daily_filled, missing_mask, sst

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
        seq_sst = []
        seq_filled = []
        seq_mask = []
        seq_sin = []
        seq_cos = []

        for date_str in date_sequence:
            try:
                _, daily_filled, missing_mask, sst = self._load_data(date_str)
                seq_filled.append(daily_filled)
                seq_mask.append(missing_mask)
                seq_sst.append(sst)
                
                # 计算时间编码
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                # 简单DOY计算
                import datetime
                doy = datetime.date(year, month, day).timetuple().tm_yday
                total_days = 366 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 365
                
                sin_val = math.sin(2 * math.pi * doy / total_days)
                cos_val = math.cos(2 * math.pi * doy / total_days)
                seq_sin.append(sin_val)
                seq_cos.append(cos_val)
                
            except Exception as e:
                return self.__getitem__((idx + 1) % len(self))

        seq_filled = np.stack(seq_filled, axis=0).astype(np.float32)  # [30, H, W]
        seq_mask = np.stack(seq_mask, axis=0).astype(np.float32)      # [30, H, W]
        seq_sst = np.stack(seq_sst, axis=0).astype(np.float32)        # [30, H, W]
        
        # 广播时间编码 [30] -> [30, H, W]
        H, W = seq_filled.shape[1], seq_filled.shape[2]
        seq_sin = np.array(seq_sin, dtype=np.float32).reshape(30, 1, 1) * np.ones((1, H, W), dtype=np.float32)
        seq_cos = np.array(seq_cos, dtype=np.float32).reshape(30, 1, 1) * np.ones((1, H, W), dtype=np.float32)

        # 目标日数据
        target_norm, target_filled, target_missing, _ = self._load_data(target_date)

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

        # 数据增强 (SST 也需要增强)
        if self.augment:
            if np.random.rand() > 0.5:
                seq_filled = np.flip(seq_filled, axis=2).copy()
                seq_mask = np.flip(seq_mask, axis=2).copy()
                seq_sst = np.flip(seq_sst, axis=2).copy()
                target_norm = np.flip(target_norm, axis=1).copy()
                artificial_mask = np.flip(artificial_mask, axis=1).copy()
                target_missing = np.flip(target_missing, axis=1).copy()
                
                lat_grid = np.flip(self.lat_norm, axis=1).copy()
                lon_grid = np.flip(self.lon_norm, axis=1).copy()
            else:
                lat_grid = self.lat_norm
                lon_grid = self.lon_norm

            if np.random.rand() > 0.5:
                seq_filled = np.flip(seq_filled, axis=1).copy()
                seq_mask = np.flip(seq_mask, axis=1).copy()
                seq_sst = np.flip(seq_sst, axis=1).copy()
                target_norm = np.flip(target_norm, axis=0).copy()
                artificial_mask = np.flip(artificial_mask, axis=0).copy()
                target_missing = np.flip(target_missing, axis=0).copy()
                
                lat_grid = np.flip(lat_grid, axis=0).copy()
                lon_grid = np.flip(lon_grid, axis=0).copy()
        else:
            lat_grid = self.lat_norm
            lon_grid = self.lon_norm

        # 组装输入 [152, H, W]
        # 0-29: SST (30)
        # 30-59: Chl-a (30)
        # 60-89: Mask (30)
        # 90: Lat (1)
        # 91: Lon (1)
        # 92-121: Sin (30)
        # 122-151: Cos (30)
        
        lat_tensor = lat_grid[np.newaxis, :, :]  # [1, H, W]
        lon_tensor = lon_grid[np.newaxis, :, :]  # [1, H, W]
        
        input_tensor = np.concatenate([
            seq_sst,      # 0-29
            seq_filled,   # 30-59
            seq_mask,     # 60-89
            lat_tensor,   # 90
            lon_tensor,   # 91
            seq_sin,      # 92-121
            seq_cos       # 122-151
        ], axis=0)

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
        sst_dir='/data_new/chla_data_imputation_data_260125/chla_data_pretraining/sst_daily_fusion_target',
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