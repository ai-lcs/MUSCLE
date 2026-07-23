# DermaMNIST-64 + ResNet18 轻量流程验证

这个目录提供一个面向初学者的 MUSCLE 两阶段实验入口。

它复用原仓库中的：

- `networks/classification/ResNet.py`
- `networks/classification/ResNet_Multi_Scale.py`
- `losses/losses.py`

没有修改论文五套数据集的读取代码，也没有修改 MUSCLE 核心算法。

## 这项实验能证明什么

实验使用 DermaMNIST-64 和 ResNet18，目标是验证：

1. DermaMNIST 固定划分能够被正确读取；
2. 普通 ResNet18 可以训练并保存 checkpoint；
3. MUSCLE 可以加载第一阶段 checkpoint；
4. 第二阶段的骨干网络被冻结；
5. MAFC 和证据网络能够得到梯度并更新；
6. 四个尺度都产生非负 evidence；
7. 能输出 Accuracy、宏平均指标、混淆矩阵和平均 uncertainty。

这只能称为：

> 在轻量代理数据与较小骨干上验证 MUSCLE 两阶段方法流程。

不能称为：

> 复现论文五套数据集上的表格结果。

## 与论文正式实验的差异

| 项目 | 论文主要设置 | 本实验 |
|---|---|---|
| 数据 | ISIC、APTOS、KvasirV2、Chaoyang、CheXpert | DermaMNIST-64 |
| 骨干 | ResNet50、VanillaNet5、Swin-T | ResNet18 |
| 输入 | 256×256 | 256×256 |
| 第一阶段 | 500 epoch | 默认 3 epoch |
| 第二阶段 | 200 epoch | 默认 3 epoch |
| 硬件 | NVIDIA RTX A6000 | CPU |
| 优化器 | SGD | SGD |
| 初始学习率 | 0.01 | 0.01 |
| momentum | 0.9 | 0.9 |
| weight decay | 0.0001 | 0.0001 |
| 学习率 | poly 策略 | poly 策略 |

论文设置核对自原文第 IV-A 节。由于训练轮数和数据规模大幅缩小，本实验指标不应与论文表格直接比较。

## 文件说明

```text
experiments/dermamnist_resnet18/
├─ dataset.py             DermaMNIST NPZ 数据读取与 256×256 变换
├─ run_smoke_test.py      单批次检查和 3+3 epoch 两阶段实验
├─ RESULTS.md             本机真实运行结果与边界
└─ README.md              当前说明
```

## 为什么先运行单批次检查

单批次检查会：

- 对 baseline 做一次前向传播、反向传播和参数更新；
- 保存并重新加载 baseline checkpoint；
- 创建 ResNet18 + MUSCLE；
- 对 MUSCLE 做一次前向传播和反向传播；
- 检查骨干梯度、融合模块梯度、evidence 形状和非负性。

只有单批次检查通过，才运行 3+3 epoch。

## 运行命令

以下路径是本机示例。数据、权重和 checkpoint 不会上传 GitHub。

### 单批次机制检查

```powershell
D:\杂文件\MUSCLE复现\.venv-cpu\Scripts\python.exe `
  experiments\dermamnist_resnet18\run_smoke_test.py `
  --data-path D:\杂文件\MUSCLE复现\downloads\dermamnist_64.npz `
  --pretrained-path D:\杂文件\MUSCLE复现\downloads\resnet18-f37072fd.pth `
  --output-dir D:\杂文件\MUSCLE复现\outputs\dermamnist_verify `
  --train-samples 32 --val-samples 14 --test-samples 14 `
  --batch-size 4 --threads 4 --verify-only
```

### 3+3 epoch 轻量训练

```powershell
D:\杂文件\MUSCLE复现\.venv-cpu\Scripts\python.exe `
  experiments\dermamnist_resnet18\run_smoke_test.py `
  --data-path D:\杂文件\MUSCLE复现\downloads\dermamnist_64.npz `
  --pretrained-path D:\杂文件\MUSCLE复现\downloads\resnet18-f37072fd.pth `
  --output-dir D:\杂文件\MUSCLE复现\outputs\dermamnist_smoke_3plus3 `
  --baseline-epochs 3 --muscle-epochs 3 `
  --train-samples 224 --val-samples 70 --test-samples 140 `
  --batch-size 8 --threads 4
```

## 输出文件

本机输出目录会包含：

```text
run_config.json
run.log
verification.json
experiment_result.json
checkpoints/
├─ baseline_best.pth
└─ muscle_best.pth
```

GitHub 只保留代码、说明和去除本机路径后的结果摘要。数据、权重、checkpoint 和完整日志只留在本机。
