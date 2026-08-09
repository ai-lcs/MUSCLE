# DermaMNIST-64 + ResNet18 两阶段轻量实验

本目录在不修改原仓库核心算法的前提下，将 DermaMNIST-64 与 ResNet18 接入 MUSCLE，用于检查两阶段训练链路在 Windows CPU 环境下能否运行。实验复用了 `ResNet.py`、`ResNet_Multi_Scale.py` 和 `losses.py`。

## 实验设计

第一阶段训练普通 ResNet18，保存并重新加载 checkpoint。第二阶段由该 checkpoint 创建 ResNet18 + MUSCLE，冻结骨干网络，只更新 MAFC 与证据分支。程序同时检查四个尺度的 evidence 形状、非负性、骨干梯度与新增模块梯度。

| 项目 | 论文主要设置 | 本实验 |
|---|---|---|
| 数据 | ISIC、APTOS、KvasirV2、Chaoyang、CheXpert | DermaMNIST-64 子集 |
| 骨干 | ResNet50、VanillaNet5、Swin-T | ResNet18 |
| 输入尺寸 | 256×256 | 256×256 |
| 第一阶段 | 500 epoch | 3 epoch |
| 第二阶段 | 200 epoch | 3 epoch |
| 硬件 | NVIDIA RTX A6000 | CPU，4 线程 |
| 优化器 | SGD | SGD |

上述差异决定了本实验只用于检查方法流程，不对应论文表格中的复现结果。

DermaMNIST 原图为 64×64。本实验将图像放大到 256×256，只是为了适配公开 MUSCLE ResNet 实现预设的阶段特征尺寸，不会增加新的图像细节。

## 文件

| 文件 | 内容 |
|---|---|
| `dataset.py` | DermaMNIST NPZ 读取、分层抽样与 256×256 图像变换 |
| `run_smoke_test.py` | 单批次机制检查与两阶段训练 |
| `RESULTS.md` | 运行环境、指标、混淆矩阵和结果解释 |
| `requirements-cpu.txt` | 本次 CPU 运行所需的最小顶层依赖版本 |
| `results/smoke_3plus3_summary.json` | 去除本机路径后的结构化结果摘要 |

## 实际环境与材料来源

本次记录使用 Python 3.11.9、PyTorch 2.11.0+cpu、torchvision 0.26.0、NumPy 1.26.4、Pillow 10.4.0 和 scikit-learn 1.8.0。`requirements-cpu.txt` 只记录该轻量入口直接需要的顶层包，不代替原仓库完整环境。

- DermaMNIST-64：[Zenodo 官方文件](https://zenodo.org/records/10519652/files/dermamnist_64.npz?download=1)，MD5 `b70a2f5635c6199aeaa28c31d7202e1f`；
- ResNet18 ImageNet-1K 权重：[PyTorch 官方文件](https://download.pytorch.org/models/resnet18-f37072fd.pth)，SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。

## 运行方式

程序需要单独准备 DermaMNIST-64 NPZ 文件和 ResNet18 预训练权重。以下命令中的路径仅作占位。

单批次机制检查：

```powershell
python experiments\dermamnist_resnet18\run_smoke_test.py `
  --data-path X:\data\dermamnist_64.npz `
  --pretrained-path X:\weights\resnet18-f37072fd.pth `
  --output-dir X:\outputs\dermamnist_verify `
  --train-samples 32 --val-samples 14 --test-samples 14 `
  --batch-size 4 --threads 4 --verify-only
```

3+3 epoch 训练：

```powershell
python experiments\dermamnist_resnet18\run_smoke_test.py `
  --data-path X:\data\dermamnist_64.npz `
  --pretrained-path X:\weights\resnet18-f37072fd.pth `
  --output-dir X:\outputs\dermamnist_smoke_3plus3 `
  --baseline-epochs 3 --muscle-epochs 3 `
  --train-samples 224 --val-samples 70 --test-samples 140 `
  --batch-size 8 --threads 4
```

运行目录会生成配置、日志、机制检查、实验结果以及两阶段 checkpoint。数据、权重、checkpoint 和完整日志未提交至 GitHub。

第一次复跑建议先使用 `--verify-only`。运行结束后检查 `verification.json` 中的 `passed`、evidence 形状、骨干梯度和参数变化；这些字段全部符合预期后，再运行 3+3 epoch。
