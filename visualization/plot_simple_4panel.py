#!/usr/bin/env python3
"""
简化版4连图可视化 - 用于152通道FNO-CBAM模型
Author: Claude
Date: 2026-02-06
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import argparse
import sys
import math
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.fno_cbam_chla import FNO_CBAM_Chla

def format_lon(x, pos):
    return f"{abs(x):.0f}°{'E' if x >= 0 else 'W'}"

def format_lat(y, pos):
    return f"{abs(y):.0f}°{'N' if y >= 0 else 'S'}"

def generate_artificial_mask(missing_mask, mask_ratio=0.2, seed=None):
    """在观测区域生成人工挖空掩码"""
    if seed is not None:
        np.random.seed(seed)

    H, W = missing_mask.shape
    observed_mask = (missing_mask == 0)
    num_observed = observed_mask.sum()

    if num_observed == 0:
        return np.zeros_like(missing_mask, dtype=np.float32)

    target_masked = int(num_observed * mask_ratio)
    artificial_mask = np.zeros((H, W), dtype=np.float32)
    current_masked = 0

    obs_rows, obs_cols = np.where(observed_mask)
    if len(obs_rows) == 0:
        return artificial_mask

    row_min, row_max = obs_rows.min(), obs_rows.max()
    col_min, col_max = obs_cols.min(), obs_cols.max()

    min_hole_size, max_hole_size = 5, 30
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
        current_masked = ((artificial_mask > 0) & observed_mask).sum()

    artificial_mask = (artificial_mask * observed_mask.astype(np.float32))
    return artificial_mask

def load_data_window(data_dir, sst_dir, target_date, window_size=30):
    """加载30天窗口数据"""
    # 收集所有可用日期
    all_dates = []
    data_path = Path(data_dir)
    for year_dir in sorted(data_path.iterdir()):
        if not year_dir.is_dir():
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            for h5_file in sorted(month_dir.glob("*.h5")):
                all_dates.append(h5_file.stem)

    # 找到目标日期的位置
    if target_date not in all_dates:
        raise ValueError(f"Target date {target_date} not found in {data_dir}")

    idx = all_dates.index(target_date)
    if idx < window_size - 1:
        raise ValueError(f"Not enough history for {target_date} (need {window_size} days, have {idx+1})")

    # 取前30天作为窗口
    date_sequence = all_dates[idx - window_size + 1:idx + 1]

    # 加载数据
    seq_sst = []
    seq_filled = []
    seq_mask = []
    seq_sin = []
    seq_cos = []

    for date_str in date_sequence:
        y = date_str[:4]
        m = date_str[4:6]

        # Load Chl-a
        chla_path = Path(data_dir) / y / m / f"{date_str}.h5"
        with h5py.File(chla_path, 'r') as f:
            daily_filled = f['daily_filled'][:]
            missing_mask = f['missing_mask'][:]

        # Load SST
        sst_path = Path(sst_dir) / y / m / f"{date_str}.h5"
        if sst_path.exists():
            with h5py.File(sst_path, 'r') as f:
                keys = list(f.keys())
                sst_key = 'daily_sst' if 'daily_sst' in keys else keys[0]
                sst = f[sst_key][:]
                # Fill NaN
                if np.isnan(sst).any():
                    ocean_mean = np.nanmean(sst)
                    if np.isnan(ocean_mean):
                        ocean_mean = 0.0
                    sst = np.nan_to_num(sst, nan=ocean_mean)
        else:
            sst = np.zeros_like(daily_filled)

        seq_sst.append(sst)
        seq_filled.append(daily_filled)
        seq_mask.append(missing_mask)

        # Time encoding
        year_int = int(date_str[:4])
        month_int = int(date_str[4:6])
        day_int = int(date_str[6:8])
        from datetime import date as dt_date
        doy = dt_date(year_int, month_int, day_int).timetuple().tm_yday
        total_days = 366 if (year_int % 4 == 0 and year_int % 100 != 0) or (year_int % 400 == 0) else 365

        sin_val = math.sin(2 * math.pi * doy / total_days)
        cos_val = math.cos(2 * math.pi * doy / total_days)
        seq_sin.append(sin_val)
        seq_cos.append(cos_val)

    # Stack
    seq_sst = np.stack(seq_sst, axis=0).astype(np.float32)
    seq_filled = np.stack(seq_filled, axis=0).astype(np.float32)
    seq_mask = np.stack(seq_mask, axis=0).astype(np.float32)

    return {
        'seq_sst': seq_sst,
        'seq_filled': seq_filled,
        'seq_mask': seq_mask,
        'seq_sin': np.array(seq_sin, dtype=np.float32),
        'seq_cos': np.array(seq_cos, dtype=np.float32),
        'target_filled': seq_filled[-1],
        'target_mask': seq_mask[-1]
    }

def run_inference(model, data, artificial_mask, device):
    """运行推理"""
    seq_sst = data['seq_sst'].copy()
    seq_filled = data['seq_filled'].copy()
    seq_mask = data['seq_mask'].copy()
    seq_sin = data['seq_sin']
    seq_cos = data['seq_cos']

    # 应用人工mask
    combined_mask = np.maximum(seq_mask[-1], artificial_mask)
    seq_mask[-1] = combined_mask
    seq_filled[-1] = np.where(artificial_mask == 1, 0.5, seq_filled[-1])

    H, W = seq_filled.shape[1], seq_filled.shape[2]

    # 广播时间编码
    seq_sin_broad = seq_sin.reshape(30, 1, 1) * np.ones((1, H, W), dtype=np.float32)
    seq_cos_broad = seq_cos.reshape(30, 1, 1) * np.ones((1, H, W), dtype=np.float32)

    # Lat/Lon grids
    lat = np.linspace(15.0, 24.0, H)
    lon = np.linspace(111.0, 118.0, W)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')

    lat_norm = (lat_grid - 19.5) / 4.5
    lon_norm = (lon_grid - 114.5) / 3.5

    lat_tensor = lat_norm[np.newaxis, :, :].astype(np.float32)
    lon_tensor = lon_norm[np.newaxis, :, :].astype(np.float32)

    # 构建152通道输入
    input_tensor = np.concatenate([
        seq_sst,       # 0-29
        seq_filled,    # 30-59
        seq_mask,      # 60-89
        lat_tensor,    # 90
        lon_tensor,    # 91
        seq_sin_broad, # 92-121
        seq_cos_broad  # 122-151
    ], axis=0)

    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)

    return pred.squeeze().cpu().numpy(), lat, lon

def create_4panel_plot(data, fno_pred, artificial_mask, lat, lon, date_str, save_path):
    """创建4连图"""
    target_filled = data['target_filled']
    target_mask = data['target_mask']

    # ================= 修改开始 =================
    # 原始逻辑：翻转数据 np.flipud(...) 和 lat[::-1] 以实现上北下南。
    # 修改逻辑：为了实现"内容上下颠倒"，我们注释掉数据的翻转，但保留 lat 的翻转。
    # 这样，原始数据 index 0 (代表 15°N/南) 将会被绘制在 lat index 0 (代表 24°N/图的顶部)。
    # 结果：南在图的上部，内容被颠倒。
    
    # target_filled = np.flipud(target_filled)  # <--- 已注释
    # target_mask = np.flipud(target_mask)      # <--- 已注释
    # fno_pred = np.flipud(fno_pred)            # <--- 已注释
    # artificial_mask = np.flipud(artificial_mask) # <--- 已注释
    
    lat = lat[::-1]  # 保留翻转lat从24到15
    # ================= 修改结束 =================

    lon_grid, lat_grid = np.meshgrid(lon, lat)

    cmap_chla = 'YlGn'
    cmap_error = 'Reds'
    missing_color = '#D3D3D3'

    valid_mask = (target_mask == 0)

    # Chla颜色范围
    valid_data = target_filled[valid_mask]
    if len(valid_data) > 0:
        vmin_chla = np.percentile(valid_data, 2)
        vmax_chla = np.percentile(valid_data, 98)
    else:
        vmin_chla, vmax_chla = 0, 1

    # 计算误差
    error = np.abs(fno_pred - target_filled)
    masked_region = (artificial_mask > 0)

    if masked_region.sum() > 0:
        mae = np.mean(error[masked_region])
        rmse = np.sqrt(np.mean(error[masked_region]**2))
    else:
        mae = rmse = 0

    total_obs = valid_mask.sum()
    mask_ratio = masked_region.sum() / total_obs * 100 if total_obs > 0 else 0
    missing_ratio = target_mask.mean() * 100

    # 创建图形
    fig = plt.figure(figsize=(24, 6))
    gs = gridspec.GridSpec(1, 6, figure=fig,
                          width_ratios=[1, 1, 1, 0.05, 1, 0.05],
                          wspace=0.12,
                          left=0.03, right=0.97, top=0.82, bottom=0.12)

    axes = [fig.add_subplot(gs[0, i]) for i in [0, 1, 2, 4]]

    # 1. Original
    ax = axes[0]
    ax.set_facecolor(missing_color)
    data_plot = np.ma.masked_where(target_mask > 0, target_filled)
    im1 = ax.pcolormesh(lon_grid, lat_grid, data_plot,
                  cmap=cmap_chla, vmin=vmin_chla, vmax=vmax_chla, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.set_title(f'Original (Missing: {missing_ratio:.1f}%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_aspect('equal')

    # 2. Input (Masked)
    ax = axes[1]
    ax.set_facecolor(missing_color)
    data_masked = target_filled.copy()
    data_masked[artificial_mask > 0] = np.nan
    data_plot = np.ma.masked_where((target_mask > 0) | (artificial_mask > 0), data_masked)
    ax.pcolormesh(lon_grid, lat_grid, data_plot,
                  cmap=cmap_chla, vmin=vmin_chla, vmax=vmax_chla, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.set_title(f'Input (Masked: {mask_ratio:.1f}%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_aspect('equal')
    ax.set_yticks([])

    # 3. FNO Prediction
    ax = axes[2]
    ax.set_facecolor(missing_color)
    fno_display = target_filled.copy()
    fno_display[artificial_mask > 0] = fno_pred[artificial_mask > 0]
    data_plot = np.ma.masked_where(target_mask > 0, fno_display)
    im3 = ax.pcolormesh(lon_grid, lat_grid, data_plot,
                        cmap=cmap_chla, vmin=vmin_chla, vmax=vmax_chla, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.set_title('FNO Prediction', fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_aspect('equal')
    ax.set_yticks([])

    # 4. Error
    ax = axes[3]
    ax.set_facecolor('white')
    error_display = error.copy()
    error_display[~masked_region] = np.nan
    data_plot = np.ma.masked_where(~masked_region, error_display)
    vmax_error = np.percentile(error[masked_region], 95) if masked_region.sum() > 0 else 0.1
    im4 = ax.pcolormesh(lon_grid, lat_grid, data_plot,
                        cmap=cmap_error, vmin=0, vmax=vmax_error, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.set_title(f'Error (MAE={mae:.4f}, RMSE={rmse:.4f})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_aspect('equal')
    ax.set_yticks([])

    # Colorbars
    cbar1_ax = fig.add_subplot(gs[0, 3])
    cbar1 = fig.colorbar(im3, cax=cbar1_ax, orientation='vertical')
    cbar1.set_label('Chl-a (mg/m³)', fontsize=10)

    cbar2_ax = fig.add_subplot(gs[0, 5])
    cbar2 = fig.colorbar(im4, cax=cbar2_ax, orientation='vertical')
    cbar2.set_label('Absolute Error', fontsize=10)

    # 总标题
    fig.suptitle(f'Chlorophyll-a Reconstruction - {date_str}',
                fontsize=14, fontweight='bold', y=0.95)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Simple 4-Panel Visualization for 152-channel FNO-CBAM')
    parser.add_argument('--date', type=str, required=True, help='Date (YYYYMMDD)')
    parser.add_argument('--data-dir', type=str, required=True, help='Chl-a data directory')
    parser.add_argument('--sst-dir', type=str, required=True, help='SST data directory')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--mask-ratio', type=float, default=0.2, help='Artificial mask ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')

    args = parser.parse_args()

    # Setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model}")
    model = FNO_CBAM_Chla(
        in_channels=152,
        out_channels=1,
        modes1=36,
        modes2=28,
        width=64,
        depth=6,
    ).to(device)

    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"Model loaded (Epoch: {checkpoint.get('epoch', 'N/A')})")

    # Load data
    print(f"Loading data for {args.date}")
    data = load_data_window(args.data_dir, args.sst_dir, args.date, window_size=30)

    # Generate artificial mask
    artificial_mask = generate_artificial_mask(
        data['target_mask'],
        mask_ratio=args.mask_ratio,
        seed=args.seed
    )

    # Run inference
    print("Running inference...")
    fno_pred, lat, lon = run_inference(model, data, artificial_mask, device)

    # Create plot
    save_path = output_dir / f"chla_4panel_{args.date}.png"
    print("Creating plot...")
    create_4panel_plot(data, fno_pred, artificial_mask, lat, lon, args.date, save_path)

    print("Done!")

if __name__ == '__main__':
    main()