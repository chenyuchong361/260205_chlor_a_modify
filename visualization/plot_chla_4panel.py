#!/usr/bin/env python3
"""
Chla FNO-CBAM 重建可视化 - 4连图 (修改版)
1. 原始数据 (真实缺失) - 移至第一位
2. 填充后+Mask图 (显示人工挖空区域) - 移至第二位
3. FNO预测结果
4. 误差图
(移除了填充后完整图 Ground Truth)

Author: Claude
Date: 2025-01-30
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
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.fno_cbam_chla import FNO_CBAM_Chla

# ============================================================================
# Configuration
# ============================================================================
# 数据路径
CROPPED_DATA_DIR = PROJECT_ROOT / 'data' / 'chla_cropped_regions_gen'
FULL_DATA_DIR = PROJECT_ROOT / 'data' / 'chla_daily_full'
MODEL_PATH = str(PROJECT_ROOT / 'checkpoints' / 'chla_pretrain_newloss_fast' / 'best_model.pth')
OUTPUT_DIR = PROJECT_ROOT / 'visualization' / 'output'

# 13个预定义区域信息 (保持不变)
REGION_INFO = {
    1: {"name": "target region", "lat": (15.0, 24.0), "lon": (111.0, 118.0)},
    2: {"name": "台湾以东", "lat": (15.0, 24.0), "lon": (120.0, 127.0)},
    3: {"name": "西太平洋", "lat": (15.0, 24.0), "lon": (130.0, 137.0)},
    4: {"name": "南海中部", "lat": (10.0, 19.0), "lon": (115.0, 122.0)},
    5: {"name": "东海", "lat": (20.0, 29.0), "lon": (125.0, 132.0)},
    6: {"name": "孟加拉湾", "lat": (5.0, 14.0), "lon": (95.0, 102.0)},
    7: {"name": "日本南部", "lat": (25.0, 34.0), "lon": (135.0, 142.0)},
    8: {"name": "赤道区", "lat": (0.0, 9.0), "lon": (110.0, 117.0)},
    9: {"name": "西太暖池", "lat": (10.0, 19.0), "lon": (140.0, 147.0)},
    10: {"name": "黑潮延伸", "lat": (20.0, 29.0), "lon": (145.0, 152.0)},
    11: {"name": "爪哇海", "lat": (-14.0, -5.0), "lon": (100.0, 107.0)},
    12: {"name": "帝汶海", "lat": (-19.0, -10.0), "lon": (115.0, 122.0)},
    13: {"name": "珊瑚海", "lat": (-24.0, -15.0), "lon": (150.0, 157.0)},
}

def setup_matplotlib():
    """配置matplotlib高质量绘图"""
    plt.rc('font', size=12)
    plt.rc('axes', linewidth=1.5, labelsize=12)
    plt.rc('lines', linewidth=1.5)
    params = {
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    }
    plt.rcParams.update(params)

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
    
    min_hole_size, max_hole_size = 5, 25
    max_attempts = 500
    
    for _ in range(max_attempts):
        if current_masked >= target_masked:
            break
            
        hole_h = np.random.randint(min_hole_size, max_hole_size + 1)
        hole_w = np.random.randint(min_hole_size, max_hole_size + 1)
        
        if row_max - hole_h < row_min or col_max - hole_w < col_min:
            continue
            
        row_start = np.random.randint(row_min, max(row_min + 1, row_max - hole_h + 1))
        col_start = np.random.randint(col_min, max(col_min + 1, col_max - hole_w + 1))
        
        artificial_mask[row_start:row_start+hole_h, col_start:col_start+hole_w] = 1
        current_masked = (artificial_mask * observed_mask).sum()
        
    return (artificial_mask * observed_mask).astype(np.float32)

def load_model(model_path, device):
    """加载FNO-CBAM模型"""
    model = FNO_CBAM_Chla(
        in_channels=152, 
        out_channels=1,
        modes1=36, 
        modes2=28,
        width=64,
        depth=6
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith('module.') else k
        new_state_dict[new_key] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'N/A')
    print(f"Model loaded (Epoch {epoch})")
    return model

def load_region_data(region_id, target_date, window_size=30):
    """加载区域数据"""
    region_dir = CROPPED_DATA_DIR / f"region_{region_id:02d}"
    
    # 收集该区域所有日期
    dates = []
    for year_dir in sorted(region_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            for h5_file in sorted(month_dir.glob("*.h5")):
                dates.append(h5_file.stem)
    
    if target_date not in dates:
        print(f"Error: Date {target_date} not found in region {region_id}")
        return None
        
    target_idx = dates.index(target_date)
    if target_idx < window_size - 1:
        print(f"Error: Not enough data before {target_date}")
        return None
        
    # 加载30天数据
    start_idx = target_idx - window_size + 1
    date_sequence = dates[start_idx:target_idx + 1]
    
    seq_filled = []
    seq_mask = []
    seq_norm = []
    
    for date_str in date_sequence:
        year = date_str[:4]
        month = date_str[4:6]
        h5_path = region_dir / year / month / f"{date_str}.h5"
        
        with h5py.File(h5_path, 'r') as f:
            seq_norm.append(f['daily_chla_norm'][:])
            seq_filled.append(f['daily_filled'][:])
            seq_mask.append(f['missing_mask'][:])
            
    # 计算区域对应的经纬度
    region_info = REGION_INFO[region_id]
    lat_range = region_info['lat']
    lon_range = region_info['lon']
    
    # 生成区域经纬度
    H, W = seq_filled[0].shape
    lat = np.linspace(lat_range[1], lat_range[0], H)  # 从高纬到低纬
    lon = np.linspace(lon_range[0], lon_range[1], W)
    
    return {
        'seq_filled': np.stack(seq_filled, axis=0),
        'seq_mask': np.stack(seq_mask, axis=0),
        'seq_norm': np.stack(seq_norm, axis=0),
        'target_filled': seq_filled[-1],
        'target_norm': seq_norm[-1],
        'target_missing': seq_mask[-1],
        'lat': lat,
        'lon': lon,
        'target_date': target_date,
        'region_id': region_id,
        'region_name': region_info['name']
    }

def run_inference(model, seq_filled, seq_mask, artificial_mask, device):
    """运行FNO推理"""
    input_filled = seq_filled.copy().astype(np.float32)
    input_mask = seq_mask.copy().astype(np.float32)
    
    # 在目标日应用人工mask
    combined_mask = np.maximum(input_mask[-1], artificial_mask)
    input_mask[-1] = combined_mask
    input_filled[-1] = np.where(artificial_mask == 1, 0.5, input_filled[-1])
    
    # 拼接输入: [30 filled + 30 mask] = 60 channels
    input_tensor = np.concatenate([input_filled, input_mask], axis=0)
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        pred = model(input_tensor)
        
    return pred.squeeze().cpu().numpy()

def create_five_panel_plot(data, fno_pred, artificial_mask, save_path):
    """
    创建4连图 (从原来的5连图修改)
    顺序: Original -> Input(Masked) -> FNO Prediction -> Error
    """
    setup_matplotlib()
    
    target_filled = data['target_filled']
    target_norm = data['target_norm']
    target_missing = data['target_missing']
    lat = data['lat']
    lon = data['lon']
    date_str = data['target_date']
    region_name = data['region_name']
    region_id = data['region_id']
    
    # 创建网格
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # 颜色设置
    cmap_chla = 'YlGn'  # 叶绿素用黄绿色系
    cmap_error = 'Reds'
    missing_color = '#D3D3D3'
    
    # 有效观测区域
    valid_mask = (target_missing == 0)
    
    # Chla颜色范围
    valid_data = target_filled[valid_mask]
    if len(valid_data) > 0:
        vmin_chla = np.percentile(valid_data, 2)
        vmax_chla = np.percentile(valid_data, 98)
    else:
        vmin_chla, vmax_chla = 0, 1
        
    # 计算误差 (只在人工挖空区域)
    error = np.abs(fno_pred - target_filled)
    masked_region = (artificial_mask > 0)
    
    if masked_region.sum() > 0:
        mae = np.mean(error[masked_region])
        rmse = np.sqrt(np.mean(error[masked_region]**2))
        max_err = np.max(error[masked_region])
    else:
        mae = rmse = max_err = 0
        
    total_obs = valid_mask.sum()
    mask_ratio = masked_region.sum() / total_obs * 100 if total_obs > 0 else 0
    missing_ratio = target_missing.mean() * 100
    
    # 创建图形 (宽度从28调整为24，因为少了一张图)
    fig = plt.figure(figsize=(24, 6))
    
    # 修改后的GridSpec: 4个主图 + 2个colorbar
    # 结构: [Original] [Input] [Prediction] [Colorbar1] [Error] [Colorbar2]
    gs = gridspec.GridSpec(1, 6, figure=fig,
                          width_ratios=[1, 1, 1, 0.05, 1, 0.05],
                          wspace=0.12,
                          left=0.03, right=0.97, top=0.82, bottom=0.12)
    
    # 获取Axes对象: 0,1,2是数据图, 4是误差图 (3和5是colorbar)
    axes = [fig.add_subplot(gs[0, i]) for i in [0, 1, 2, 4]]
    
    # ===== 1. 原始缺失情况 (Original) [原位置3，现位置1] =====
    ax = axes[0]
    ax.set_facecolor(missing_color)
    
    # 显示缺失分布
    data_plot = np.ma.masked_where(target_missing > 0, target_filled)
    im1 = ax.pcolormesh(lon_grid, lat_grid, data_plot, 
                  cmap=cmap_chla, vmin=vmin_chla, vmax=vmax_chla, shading='auto')
    
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat)) # 加上Y轴标签
    ax.set_title(f'Original (Missing: {missing_ratio:.1f}%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10) # 加上Y轴标签
    ax.set_aspect('equal')
    
    # ===== 2. 填充后+Mask图 (Input) [原位置2，现位置2] =====
    ax = axes[1]
    ax.set_facecolor(missing_color)
    
    data_masked = target_filled.copy()
    data_masked[artificial_mask > 0] = np.nan
    data_plot = np.ma.masked_where((target_missing > 0) | (artificial_mask > 0), data_masked)
    
    ax.pcolormesh(lon_grid, lat_grid, data_plot, 
                  cmap=cmap_chla, vmin=vmin_chla, vmax=vmax_chla, shading='auto')
    
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.set_title(f'Input (Masked: {mask_ratio:.1f}%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_aspect('equal')
    ax.set_yticks([]) # 移除Y轴标签
    
    # ===== 3. FNO预测 (Prediction) [位置3] =====
    ax = axes[2]
    ax.set_facecolor(missing_color)
    
    # FNO预测结果（在人工mask区域用预测值，其他用原值）
    fno_display = target_filled.copy()
    fno_display[artificial_mask > 0] = fno_pred[artificial_mask > 0]
    data_plot = np.ma.masked_where(target_missing > 0, fno_display)
    
    im3 = ax.pcolormesh(lon_grid, lat_grid, data_plot, 
                        cmap=cmap_chla, vmin=vmin_chla, vmax=vmax_chla, shading='auto')
    
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.set_title('FNO Prediction', fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_aspect('equal')
    ax.set_yticks([])
    
    # Chla Colorbar (放在第3列，即Prediction后面)
    cax_chla = fig.add_subplot(gs[0, 3])
    cbar_chla = plt.colorbar(im3, cax=cax_chla)
    cbar_chla.set_label('Chla (normalized)', fontsize=10)
    
    # ===== 4. 误差图 (Error) [位置4] =====
    ax = axes[3]
    ax.set_facecolor('white')
    
    error_display = np.where(artificial_mask > 0, error, np.nan)
    error_plot = np.ma.masked_where(artificial_mask == 0, error_display)
    
    # 误差范围
    vmax_err = min(0.15, max_err * 1.1) if max_err > 0 else 0.1
    
    im4 = ax.pcolormesh(lon_grid, lat_grid, error_plot, 
                        cmap=cmap_error, vmin=0, vmax=vmax_err, shading='auto')
    
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.set_title('|Error| (Masked Region)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_aspect('equal')
    ax.set_yticks([])
    
    # 统计信息
    stats_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nMax: {max_err:.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
           
    # Error Colorbar
    cax_err = fig.add_subplot(gs[0, 5])
    cbar_err = plt.colorbar(im4, cax=cax_err)
    cbar_err.set_label('|Error|', fontsize=10)
    
    # 图例
    legend_elements = [
        Patch(facecolor=missing_color, edgecolor='black', label='Missing'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', 
              bbox_to_anchor=(0.98, 0.95), fontsize=9)
    
    # 总标题
    date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    fig.suptitle(f'Chla Reconstruction: Region {region_id} ({region_name}) - {date_formatted}', 
                fontsize=14, fontweight='bold', y=0.96)
    
    # 保存
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return mae, rmse, max_err

def main():
    global CROPPED_DATA_DIR
    parser = argparse.ArgumentParser(description='Chla FNO-CBAM 4-Panel Visualization')
    parser.add_argument('--date', type=str, default='20200715', 
                        help='Target date in YYYYMMDD format')
    parser.add_argument('--region', type=int, default=1, 
                        help='Region ID (1-13)')
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU ID')
    parser.add_argument('--mask-ratio', type=float, default=0.2, 
                        help='Random mask ratio')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output file path')
    parser.add_argument('--model', type=str, default=MODEL_PATH, 
                        help='Model checkpoint path')
    parser.add_argument('--cropped-data-dir', type=str, default=str(CROPPED_DATA_DIR),
                        help='Cropped regions data directory')

    args = parser.parse_args()
    CROPPED_DATA_DIR = Path(args.cropped_data_dir)
    
    print("=" * 70)
    print("Chla FNO-CBAM Reconstruction - 4 Panel Visualization (Original, Input, Pred, Error)")
    print("=" * 70)
    print(f"Region: {args.region} ({REGION_INFO[args.region]['name']})")
    print(f"Date:   {args.date}")
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 加载模型
    print("\nLoading model...")
    model = load_model(args.model, device)
    
    # 加载数据
    print(f"\nLoading region {args.region} data...")
    data = load_region_data(args.region, args.date)
    
    if data is None:
        return
        
    print(f"  Shape: {data['seq_filled'].shape}")
    print(f"  Missing ratio: {data['target_missing'].mean()*100:.1f}%")
    
    # 生成人工mask
    print(f"\nGenerating artificial mask (ratio={args.mask_ratio})...")
    artificial_mask = generate_artificial_mask(
        data['target_missing'], 
        mask_ratio=args.mask_ratio,
        seed=args.seed
    )
    actual_ratio = artificial_mask.sum() / (data['target_missing'] == 0).sum() * 100
    print(f"  Actual mask ratio: {actual_ratio:.1f}%")
    
    # 运行推理
    print("\nRunning FNO inference...")
    fno_pred = run_inference(
        model, 
        data['seq_filled'], 
        data['seq_mask'], 
        artificial_mask, 
        device
    )
    print(f"  Prediction range: [{fno_pred.min():.4f}, {fno_pred.max():.4f}]")
    
    # 创建可视化
    if args.output:
        save_path = args.output
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = OUTPUT_DIR / f'chla_4panel_region{args.region:02d}_{args.date}.png'
        
    print(f"\nCreating visualization...")
    mae, rmse, max_err = create_five_panel_plot(data, fno_pred, artificial_mask, save_path)
    
    print(f"\nResults:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Max:  {max_err:.4f}")
    print(f"\nSaved: {save_path}")

if __name__ == '__main__':
    main()
