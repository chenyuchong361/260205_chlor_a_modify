"""
FNO_CBAM with Temporal (30-day) Input for Chla
支持60通道输入：30天Chla + 30天mask
修改版：动态尺寸支持
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv2d_fast(nn.Module):
    """快速傅里叶卷积层"""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

    def compl_mul2d(self, input, weights):
        weights_complex = torch.view_as_complex(weights)
        return torch.einsum("bixy,ioxy->boxy", input, weights_complex)

    def forward(self, x):
        batchsize = x.shape[0]
        H, W = x.size(-2), x.size(-1)
        orig_dtype = x.dtype

        # cuFFT在半精度下只支持2的幂次方，强制使用fp32进行FFT
        x_fp32 = x.float()
        x_ft = torch.fft.rfft2(x_fp32)

        out_ft = torch.zeros(batchsize, self.out_channels, H, W//2 + 1,
                            dtype=torch.cfloat, device=x.device)

        # 确保modes不超过实际尺寸
        modes1 = min(self.modes1, H // 2)
        modes2 = min(self.modes2, W // 2 + 1)

        out_ft[:, :, :modes1, :modes2] = \
            self.compl_mul2d(x_ft[:, :, :modes1, :modes2], self.weights1[:, :, :modes1, :modes2])
        out_ft[:, :, -modes1:, :modes2] = \
            self.compl_mul2d(x_ft[:, :, -modes1:, :modes2], self.weights2[:, :, :modes1, :modes2])

        x = torch.fft.irfft2(out_ft, s=(H, W))

        # 恢复原始dtype
        return x.to(orig_dtype)


class CBAM_Block(nn.Module):
    """CBAM注意力模块"""
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM_Block, self).__init__()
        reduced_channels = max(channels // reduction_ratio, 4)
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        batch_size, c, h, w = x.shape

        # Channel attention
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(batch_size, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(batch_size, c)

        avg_out = self.fc2(F.relu(self.fc1(avg_pool)))
        max_out = self.fc2(F.relu(self.fc1(max_pool)))
        channel_attention = torch.sigmoid(avg_out + max_out).view(batch_size, c, 1, 1)
        x = x * channel_attention

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = torch.sigmoid(self.conv_spatial(spatial_input))
        x = x * spatial_attention

        return x


class FeatureEngineering(nn.Module):
    """
    Feature Engineering Module
    Computes:
    1. sst_anom: SST Anomaly (SST - temporal mean)
    2. |∇sst|: SST Gradient Magnitude
    3. Δsst: SST Laplacian
    """
    def __init__(self):
        super(FeatureEngineering, self).__init__()
        # Sobel kernels
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        # Laplacian kernel
        self.register_buffer('laplacian', torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, sst):
        """
        Args:
            sst: [B, T, H, W]
        Returns:
            sst_anom, sst_grad, sst_lap (all [B, T, H, W])
        """
        B, T, H, W = sst.shape
        # Reshape to apply 2D conv to each time step
        sst_reshaped = sst.reshape(B * T, 1, H, W)

        # Gradient
        grad_x = F.conv2d(sst_reshaped, self.sobel_x, padding=1)
        grad_y = F.conv2d(sst_reshaped, self.sobel_y, padding=1)
        sst_grad = torch.sqrt(grad_x**2 + grad_y**2).reshape(B, T, H, W)

        # Laplacian
        sst_lap = F.conv2d(sst_reshaped, self.laplacian, padding=1).reshape(B, T, H, W)

        # Anomaly: Subtract sliding window mean (temporal mean)
        # Represents deviation from the 30-day average state
        sst_mean = sst.mean(dim=1, keepdim=True)
        sst_anom = sst - sst_mean

        return sst_anom, sst_grad, sst_lap


class FNO_CBAM_Chla(nn.Module):
    """
    FNO with CBAM for Chla reconstruction with Multi-source Input

    Input: [B, 152, H, W] (Default assumption)
      - 00-29: SST (30)
      - 30-59: Chl-a (30)
      - 60-89: Mask (30)
      - 90-90: Lat (1)
      - 91-91: Lon (1)
      - 92-121: Day Sin (30)
      - 122-151: Day Cos (30)

    Feature Engineering adds 90 channels:
      - sst_anom (30)
      - sst_grad (30)
      - sst_lap (30)

    Total Internal Channels: 152 + 90 = 242

    Output: [B, 1, H, W] (Reconstructed Chla at last step)
    """
    def __init__(
        self,
        in_channels=152,
        out_channels=1,
        modes1=36,
        modes2=28,
        width=64,
        depth=6,
        cbam_reduction_ratio=16,
        sst_channels=30,
    ):
        super(FNO_CBAM_Chla, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.depth = depth
        self.sst_channels = sst_channels

        # Feature Engineering Module
        self.feature_eng = FeatureEngineering()

        # Calculate total input channels after feature engineering
        # 3 derived features per SST channel
        self.aug_channels = in_channels + sst_channels * 3

        # Input Projection (Unified)
        self.input_proj = nn.Sequential(
            nn.Conv2d(self.aug_channels, width, kernel_size=1),
            nn.GELU(),
        )

        # FNO层
        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()
        self.cbams = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(self.depth):
            self.convs.append(SpectralConv2d_fast(width, width, modes1, modes2))
            self.ws.append(nn.Conv2d(width, width, 1))
            self.cbams.append(CBAM_Block(channels=width, reduction_ratio=cbam_reduction_ratio))
            # 使用GroupNorm代替LayerNorm，不依赖固定尺寸
            self.norms.append(nn.GroupNorm(num_groups=8, num_channels=width))

        # 输出解码
        self.decoder = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

    def forward(self, x):
        """
        Args:
            x: [B, 152, H, W] - Combined features

        Returns:
            pred: [B, 1, H, W] - 第30天的重建值
        """
        # Extract SST for feature engineering
        # Assumption: SST is at the beginning [0:30]
        sst = x[:, :self.sst_channels, :, :]

        # Compute derived features
        sst_anom, sst_grad, sst_lap = self.feature_eng(sst)

        # Concatenate all features
        x_aug = torch.cat([x, sst_anom, sst_grad, sst_lap], dim=1)

        # Project to hidden dim
        x = self.input_proj(x_aug)

        # FNO处理
        for i in range(self.depth):
            residual = x

            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2

            x = self.cbams[i](x)
            x = self.norms[i](x)

            x = residual + x
            x = F.gelu(x)

        # 解码输出
        pred = self.decoder(x)  # [B, 1, H, W]

        return pred



def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("FNO_CBAM_Chla 模型测试")
    print("=" * 60)

    # 创建模型 - 目标域尺寸 181x141
    model = FNO_CBAM_Chla(
        in_channels=60,
        out_channels=1,
        modes1=36,  # 约 H/5
        modes2=28,  # 约 W/5
        width=64,
        depth=6,
    ).to(device)

    print(f"\n模型参数量: {count_parameters(model) / 1e6:.2f}M")

    # 测试不同尺寸
    test_sizes = [(181, 141), (256, 256), (128, 128)]

    for H, W in test_sizes:
        print(f"\n测试尺寸: ({H}, {W})")

        B = 2
        x = torch.randn(B, 60, H, W).to(device)

        with torch.no_grad():
            output = model(x)

        print(f"  输入: {x.shape}")
        print(f"  输出: {output.shape}")
        print(f"  输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")

    print("\n" + "=" * 60)
    print("✓ 模型测试通过")
    print("=" * 60)
