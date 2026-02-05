# 重新训练计划（新模型输入 + 新损失）

## 1. 目标与背景

### 核心目标
在叶绿素日尺度缺失填充任务中，引入多源输入特征和 SST 特征工程，替换原有的简单双通道输入架构，同时简化损失函数为纯 Masked MAE。

### 关键变更
- **输入升级**：从 60 通道扩展至 152 通道（SST/Chl-a/Mask/坐标/时间编码）
- **特征工程**：模型内部基于 SST 自动生成距平、梯度、拉普拉斯算子（90 通道），最终主干输入 242 通道
- **损失简化**：移除所有辅助损失，仅保留在人工挖空区域计算的 Masked MAE

---

## 2. 架构设计

### 2.1 输入通道约定（152 通道）

**通道顺序必须严格遵循以下约定**（模型与数据加载器需保持一致）：

```
Index   | Channels | Feature          | Notes
--------|----------|------------------|------------------------
0-29    | 30       | SST              | 30天窗口，JAXA SST
30-59   | 30       | Chl-a (masked)   | 目标日前30天，缺失处已填充
60-89   | 30       | Missing Mask     | 1=缺失, 0=观测
90      | 1        | Latitude         | 归一化坐标网格
91      | 1        | Longitude        | 归一化坐标网格
92-121  | 30       | sin(day_of_year) | 按H×W广播
122-151 | 30       | cos(day_of_year) | 按H×W广播
```

### 2.2 模型内部特征工程

模型在 forward 时自动从 SST（通道 0-29）生成：
- `sst_anom`（30 通道）：SST - mean(SST)
- `∇sst`（30 通道）：梯度幅值 √(∂sst/∂x)² + (∂sst/∂y)²
- `Δsst`（30 通道）：拉普拉斯算子 ∂²sst/∂x² + ∂²sst/∂y²

这 90 通道特征与原始 152 通道拼接后进入主干网络（FNO + CBAM），总计 242 通道。

### 2.3 损失函数

**仅使用 Masked MAE**：
```python
loss = sum(|pred - target| * artificial_mask) / (artificial_mask.sum() + eps)
```

- `artificial_mask`：训练时人工挖空的区域（1=需要预测的缺失位置）
- 不在观测区域计算损失，避免模型学习简单的复制策略

---

## 3. 数据准备清单

### 3.1 必需数据
- [ ] **Chl-a**：已预处理好的数据
- [ ] **Missing mask**：原始观测缺失标记（1=缺失，0=有观测）
- [ ] **JAXA SST**：30 天窗口，与 Chl-a 时间对齐
- [ ] **Lat/Lon grids**：归一化至 [-1, 1] 或 [0, 1]
- [ ] **Day-of-year encoding**：每个时间步计算 sin/cos 编码

### 3.2 数据质量检查
在训练前务必验证：
1. 所有数据时间对齐（SST、Chl-a、时间编码）
2. 归一化范围正确（Chl-a 在 [0,1]，坐标在 [-1,1] 或 [0,1]）
3. Mask 语义一致（1=缺失在整个管道中保持不变）
4. 无 NaN/Inf 值（或已正确处理）

---

## 4. 代码改造路径（检查是否符合新模型输入和新损失函数）

### 4.1 模型改造（`models/fno_cbam_chla.py`）
- [ ] 修改 `__init__` 中 `in_channels=152`
- [ ] 在 `forward` 中实现 SST 特征工程模块（距平/梯度/拉普拉斯）
- [ ] 验证内部通道拼接逻辑：152 + 90 = 242

### 4.2 损失改造（`losses/chla_loss.py`）
- [ ] 移除所有辅助损失项（SSIM、边缘损失、时间一致性等）
- [ ] 保留 `masked_mae(pred, target, artificial_mask)`
- [ ] 确保 `artificial_mask` 作为必需参数传入

### 4.3 数据加载器改造（`datasets/*.py`）
- [ ] 输出 `batch['input']` 为 `[B, 152, H, W]`（按 2.1 节通道顺序）
- [ ] 输出 `batch['target']` 为 `[B, H, W]` 或 `[B, 1, H, W]`
- [ ] 输出 `batch['artificial_mask']`（训练时人工挖空的位置）
- [ ] 输出 `batch['original_mask']`（可选，用于后处理）

### 4.4 训练脚本改造（`train.py`）
- [ ] 移除 baseline 相关代码（如 `inputs[:, 29, :, :]`）
- [ ] 调整损失计算为 `loss = loss_fn(pred, target, artificial_mask)`
- [ ] 更新日志记录：仅输出 `train_loss` / `val_loss`

---

## 5. 训练策略（分阶段验证）

### Phase 1: 形状与数值检查
**目标**：验证数据管道和模型前向传播无错误


**检查项**：
- [ ] 输入通道顺序与模型假设一致
- [ ] Chl-a 归一化范围在 [0, 1]（允许 ±0.05 越界后 clip）
- [ ] Mask 为二值 {0, 1}
- [ ] 时间编码 sin/cos 在 [-1, 1]
- [ ] 模型前向传播无 NaN/Inf

### Phase 2: 快速测试
**目标**：验证模型是否可以跑通

使用10个epoch，lr=1e-3, batch_size=8进行快速训练

**通过标准**：可以跑通实验，10个epoch内训练有损失

### Phase 3: 完整训练（预计数小时至数天）
**配置建议**：
```yaml
model:
  in_channels: 152
  modes: 12          # 或根据实验调整
  width: 64          # 或根据实验调整
  depth: 4           # 或根据实验调整

training:
  epochs: 100
  batch_size: 8      # 根据显存调整
  lr: 1e-4
  weight_decay: 1e-5
  scheduler: CosineAnnealingLR
  amp: true          # 混合精度训练
  grad_clip: 1.0

data:
  mask_ratio: 0.3    # 训练时挖空比例
  val_mask_ratio: 0.3
```

**监控指标**：
- [ ] `train_loss` 稳定下降（无明显震荡）
- [ ] `val_loss` 在前期跟随训练损失下降
- [ ] 无异常峰值（可能指示数据或超参问题）


---

## 6. 推理与验证

### 6.1 Output Composition（关键步骤）
推理时需要将预测结果与原始观测合成：
```python
output = torch.where(original_mask == 1, prediction, observation)
```

- `original_mask == 1`：原始缺失位置（用预测值填充）
- `original_mask == 0`：有观测位置（保留原始值）

### 6.2 可视化 Checklist
为目标区域随机抽取 5-10 天，生成以下对比图：

**4-Panel 图**：
1. **Input Chl-a (masked)**：输入的填充版本
2. **Ground Truth**：目标日归一化 Chl-a
3. **Model Prediction**：模型输出（仅缺失区域）
4. **Final Output**：合成后的完整结果


**检查要点**：
- [ ] 观测区域保持不变（像素级对比）
- [ ] 缺失区域的预测在空间上连续、无异常斑块
- [ ] 预测值范围在合理区间（如 [0, 1] 或经反归一化后的物理量纲）

---

## 7. 实验记录

checkoint,output,experiment等文件夹中可以尝试保存.txt日志信息

---

## 8. 运行命令

修改或使用scripts中的脚本进行实验全流程

---

## 9. 风险与缓解策略

| 风险 | 可能原因 | 缓解策略 |
|------|---------|---------|
| 训练损失不下降 | 通道顺序错误 / 归一化问题 | 先跑 Phase 1 检查；打印输入统计量 |
| 显存溢出 | 242 通道输入 + 大分辨率 | 减小 batch_size / 使用梯度累积 / 启用 AMP |
| 过拟合 | 数据量不足 / 模型过大 | 增加 weight_decay / 使用 Dropout / 数据增强 |
| 预测区域出现异常斑块 | 损失函数覆盖不足 | 检查 artificial_mask 生成逻辑；考虑增加空间平滑损失 |
| 观测区域被错误修改 | Output composition 逻辑错误 | 断言推理时观测区域像素级不变 |




## 最后检查清单（正式训练前必须全部勾选）

- [ ] 模型 `in_channels=152` 已更新
- [ ] 数据加载器输出 152 通道（通道顺序与文档一致）
- [ ] 损失函数仅保留 Masked MAE
- [ ] Phase 1 形状检查通过
- [ ] Phase 2 快速验证测试通过
- [ ] 配置文件与模型/数据代码一致
- [ ] 实验日志记录



**当所有检查项通过后，即可开始 Phase 3 完整训练。**