#!/usr/bin/env python3
"""
叶绿素缺失值重建推理脚本

功能：
- 加载训练好的模型
- 对目标域全时段进行推理
- 输出填补后的daily Chla产品

Author: Claude
Date: 2025-01-30
"""

import os
import sys
import argparse
import numpy as np
import h5py
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fno_cbam_chla import FNO_CBAM_Chla


# 归一化参数
LN_OFFSET = 4.61
LN_SCALE = 9.22
CHLA_MIN = 0.01
CHLA_MAX = 100.0


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


def normalized_to_chla(c_n):
    """归一化值 → 叶绿素浓度 (mg/m³)"""
    ln_c = LN_SCALE * c_n - LN_OFFSET
    chla = np.exp(ln_c)
    chla = np.clip(chla, CHLA_MIN, CHLA_MAX)
    return chla


def load_model(checkpoint_path, device, args):
    """加载模型"""
    model = FNO_CBAM_Chla(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        modes1=args.modes1,
        modes2=args.modes2,
        width=args.width,
        depth=args.depth,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")

    return model


def collect_dates(data_dir):
    """收集所有可用日期"""
    dates = []
    for h5_file in sorted(Path(data_dir).rglob("*.h5")):
        dates.append(h5_file.stem)
    return dates


def _load_lat_lon_from_any_h5(data_dir: str):
    for h5_path in sorted(Path(data_dir).rglob("*.h5")):
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'latitude' in f and 'longitude' in f:
                    lat = f['latitude'][:]
                    lon = f['longitude'][:]
                    if lat is not None and lon is not None:
                        return lat, lon
        except Exception:
            continue
    return None, None


def _make_index_lat_lon(data_dir: str):
    for h5_path in sorted(Path(data_dir).rglob("*.h5")):
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'daily_chla_norm' in f:
                    h, w = f['daily_chla_norm'].shape
                    return np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32)
        except Exception:
            continue
    raise RuntimeError(f"No readable .h5 found under {data_dir}")


def resolve_lat_lon(data_dir: str):
    candidates = [data_dir]
    if data_dir.endswith('_gen'):
        candidates.append(data_dir[:-4])

    for cand in candidates:
        if not Path(cand).exists():
            continue
        lat, lon = _load_lat_lon_from_any_h5(cand)
        if lat is not None and lon is not None:
            return lat, lon

    return _make_index_lat_lon(data_dir)


def load_daily_data(data_dir, sst_dir, date_str, fallback_lat=None, fallback_lon=None):
    """加载单天数据（Chl-a和SST）"""
    year = date_str[:4]
    month = date_str[4:6]
    h5_path = Path(data_dir) / year / month / f"{date_str}.h5"
    sst_path = Path(sst_dir) / year / month / f"{date_str}.h5"

    # Load Chl-a data
    with h5py.File(h5_path, 'r') as f:
        daily_norm = f['daily_chla_norm'][:]
        daily_filled = f['daily_filled'][:]
        missing_mask = f['missing_mask'][:]
        lat = f['latitude'][:] if 'latitude' in f else fallback_lat
        lon = f['longitude'][:] if 'longitude' in f else fallback_lon

    # Load SST data
    if sst_path.exists():
        with h5py.File(sst_path, 'r') as f:
            keys = list(f.keys())
            sst_key = 'daily_sst' if 'daily_sst' in keys else keys[0]
            sst = f[sst_key][:]
            # Fill NaN values with ocean mean
            if np.isnan(sst).any():
                ocean_mean = np.nanmean(sst)
                if np.isnan(ocean_mean):
                    ocean_mean = 0.0
                sst = np.nan_to_num(sst, nan=ocean_mean)
    else:
        # If SST file missing, use zeros
        sst = np.zeros_like(daily_filled)

    return daily_norm, daily_filled, missing_mask, sst, lat, lon


def inference_single_day(
    model,
    data_dir,
    sst_dir,
    target_date,
    available_dates,
    window_size,
    device,
    fallback_lat,
    fallback_lon,
):
    """对单天进行推理"""
    import math

    # 获取日期序列
    idx = available_dates.index(target_date)
    start_idx = idx - window_size + 1

    if start_idx < 0:
        return None, None, None, None, None

    date_sequence = available_dates[start_idx:idx + 1]

    # 加载30天数据
    seq_sst = []
    seq_filled = []
    seq_mask = []
    seq_sin = []
    seq_cos = []

    for date_str in date_sequence:
        _, daily_filled, missing_mask, sst, _, _ = load_daily_data(
            data_dir,
            sst_dir,
            date_str,
            fallback_lat=fallback_lat,
            fallback_lon=fallback_lon,
        )
        seq_sst.append(sst)
        seq_filled.append(daily_filled)
        seq_mask.append(missing_mask)

        # 计算时间编码
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        from datetime import date as dt_date
        doy = dt_date(year, month, day).timetuple().tm_yday
        total_days = 366 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 365

        sin_val = math.sin(2 * math.pi * doy / total_days)
        cos_val = math.cos(2 * math.pi * doy / total_days)
        seq_sin.append(sin_val)
        seq_cos.append(cos_val)

    # Stack
    seq_sst = np.stack(seq_sst, axis=0).astype(np.float32)      # [30, H, W]
    seq_filled = np.stack(seq_filled, axis=0).astype(np.float32)  # [30, H, W]
    seq_mask = np.stack(seq_mask, axis=0).astype(np.float32)      # [30, H, W]

    H, W = seq_filled.shape[1], seq_filled.shape[2]

    # 广播时间编码 [30] -> [30, H, W]
    seq_sin = np.array(seq_sin, dtype=np.float32).reshape(30, 1, 1) * np.ones((1, H, W), dtype=np.float32)
    seq_cos = np.array(seq_cos, dtype=np.float32).reshape(30, 1, 1) * np.ones((1, H, W), dtype=np.float32)

    # 获取lat/lon网格（归一化到[-1, 1]）
    lat = np.linspace(15.0, 24.0, H)
    lon = np.linspace(111.0, 118.0, W)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')

    # 归一化
    lat_norm = (lat_grid - 19.5) / 4.5  # (x - mid) / half_range
    lon_norm = (lon_grid - 114.5) / 3.5

    lat_tensor = lat_norm[np.newaxis, :, :].astype(np.float32)  # [1, H, W]
    lon_tensor = lon_norm[np.newaxis, :, :].astype(np.float32)  # [1, H, W]

    # 构建输入 [152, H, W]
    # 0-29: SST
    # 30-59: Chl-a
    # 60-89: Mask
    # 90: Lat
    # 91: Lon
    # 92-121: Sin
    # 122-151: Cos
    input_tensor = np.concatenate([
        seq_sst,      # 0-29
        seq_filled,   # 30-59
        seq_mask,     # 60-89
        lat_tensor,   # 90
        lon_tensor,   # 91
        seq_sin,      # 92-121
        seq_cos       # 122-151
    ], axis=0)  # [152, H, W]

    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(device)  # [1, 152, H, W]

    # 推理
    with torch.no_grad():
        pred = model(input_tensor)  # [1, 1, H, W]

    pred = pred.squeeze().cpu().numpy()  # [H, W]

    # 获取目标日的原始数据和掩码
    target_norm, target_filled, target_missing, _, lat, lon = load_daily_data(
        data_dir,
        sst_dir,
        target_date,
        fallback_lat=fallback_lat,
        fallback_lon=fallback_lon,
    )

    # Output Composition: 观测保留，缺失用预测
    output_norm = np.where(target_missing == 1, pred, target_norm)

    # Clip到[0, 1]
    output_norm = np.clip(output_norm, 0, 1)

    # 反归一化
    output_chla = normalized_to_chla(output_norm)

    return output_chla, output_norm, target_missing, lat, lon


def save_result_nc(output_path, chla, lat, lon, date_str, missing_mask):
    """保存结果为NetCDF格式"""
    ds = xr.Dataset(
        {
            'chlor_a': (['latitude', 'longitude'], chla.astype(np.float32)),
            'missing_mask': (['latitude', 'longitude'], missing_mask.astype(np.int8)),
        },
        coords={
            'latitude': lat,
            'longitude': lon,
        },
        attrs={
            'title': 'Reconstructed Chlorophyll-a Concentration',
            'date': date_str,
            'units': 'mg/m^3',
            'created_by': 'FNO-CBAM Model',
        }
    )

    ds.to_netcdf(output_path)


def main():
    parser = argparse.ArgumentParser(description='Chla Inference')

    parser.add_argument('--data_dir', type=str, required=True,
                       help='Filled data directory')
    parser.add_argument('--sst_dir', type=str, required=True,
                       help='SST data directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Model checkpoint path')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--year_start', type=int, default=2015)
    parser.add_argument('--year_end', type=int, default=2025)
    parser.add_argument('--window_size', type=int, default=30)

    # 模型参数
    parser.add_argument('--in_channels', type=int, default=152)
    parser.add_argument('--out_channels', type=int, default=1)
    parser.add_argument('--modes1', type=int, default=36)
    parser.add_argument('--modes2', type=int, default=28)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--depth', type=int, default=6)

    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()

    # 创建输出目录和日志
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 创建日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(args.output_dir) / f"inference_log_{timestamp}.txt"
    logger = Logger(log_file)
    sys.stdout = logger

    print("=" * 60)
    print("Chla Inference")
    print("=" * 60)
    print(f"Timestamp: {timestamp}")
    print(f"Log file: {log_file}")
    print(f"Data dir: {args.data_dir}")
    print(f"SST dir: {args.sst_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print(f"Years: {args.year_start} - {args.year_end}")
    print("=" * 60)

    # 设备
    device = torch.device(args.device)

    # 加载模型
    model = load_model(args.checkpoint, device, args)

    lat, lon = resolve_lat_lon(args.data_dir)

    # 收集日期
    available_dates = collect_dates(args.data_dir)
    print(f"Found {len(available_dates)} available dates")

    # 过滤年份
    target_dates = [d for d in available_dates
                   if args.year_start <= int(d[:4]) <= args.year_end]

    # 跳过前window_size-1天（没有足够历史）
    valid_dates = target_dates[args.window_size - 1:]
    print(f"Processing {len(valid_dates)} dates")

    # 推理
    for date_str in tqdm(valid_dates, desc="Inferencing"):
        try:
            output_chla, output_norm, missing_mask, lat, lon = inference_single_day(
                model, args.data_dir, args.sst_dir, date_str, available_dates,
                args.window_size, device, lat, lon
            )

            if output_chla is None:
                continue

            # 保存结果
            year = date_str[:4]
            month = date_str[4:6]
            out_path = Path(args.output_dir) / year / month
            out_path.mkdir(parents=True, exist_ok=True)

            nc_file = out_path / f"{date_str}_filled.nc"
            save_result_nc(nc_file, output_chla, lat, lon, date_str, missing_mask)

        except Exception as e:
            print(f"Error processing {date_str}: {e}")
            continue

    print("\n" + "=" * 60)
    print("Inference completed!")
    print("=" * 60)

    # 关闭日志
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()
        sys.stdout = sys.__stdout__


if __name__ == '__main__':
    main()
