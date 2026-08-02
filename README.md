# MUSCLE：论文阅读与轻量实验记录

本仓库 fork 自 [Q4CS/MUSCLE](https://github.com/Q4CS/MUSCLE)，对应论文 *MUSCLE: A New Perspective to Multi-Scale Fusion for Medical Image Classification Based on the Theory of Evidence*。在保留原始实现的基础上，我补充了中文方法笔记，并用 DermaMNIST-64 和 ResNet18 进行了一次小规模两阶段实验。

## 方法概述

MUSCLE 用于医学图像分类。它从骨干网络的多个阶段提取特征，通过 Multi-Axis Feature Compression（MAFC）分别压缩通道、高度和宽度方向的信息，再为各尺度生成分类证据。第二阶段加载已经训练好的骨干网络并冻结其参数，只更新多尺度压缩与证据融合模块。

```text
输入图像
  → 骨干网络的多阶段特征
  → MAFC 三轴特征压缩
  → 各尺度分类证据
  → 证据聚合与不确定性估计
  → 分类结果
```

原始代码包含 ResNet、VanillaNet 和 Swin Transformer 三类骨干网络，并提供 ISIC 2018、APTOS 2019、KvasirV2、Chaoyang 与 CheXpert 的数据读取代码。

## 仓库内容

| 路径 | 内容 |
|---|---|
| `datasets/` | 五套医学图像数据集的读取与增强 |
| `networks/` | 骨干网络及其 MUSCLE 版本 |
| `losses/` | 证据分类损失与尺度一致性损失 |
| `main.py` | 训练、验证和测试入口 |
| [`docs/代码结构.md`](docs/代码结构.md) | 代码模块与调用关系 |
| [`docs/论文方法与代码对应.md`](docs/论文方法与代码对应.md) | 论文概念在代码中的位置 |
| [`docs/实验范围与设置.md`](docs/实验范围与设置.md) | 论文设置与轻量实验的差异 |
| [`experiments/dermamnist_resnet18/`](experiments/dermamnist_resnet18/README.md) | DermaMNIST-64 + ResNet18 实验代码与记录 |

## 轻量实验结果

轻量实验采用固定随机种子，从 DermaMNIST-64 中抽取 224 个训练样本、70 个验证样本和 140 个测试样本，在 CPU 上分别训练 ResNet18 baseline 与冻结骨干的 MUSCLE 各 3 个 epoch。实验验证了 checkpoint 加载、骨干冻结、四尺度非负 evidence 以及融合模块参数更新。

| 模型 | Accuracy | Macro-F1 |
|---|---:|---:|
| ResNet18 baseline | 0.6500 | 0.1665 |
| ResNet18 + MUSCLE | 0.6714 | 0.1148 |

MUSCLE 在该设置下将测试样本全部预测为多数类。Accuracy 的小幅上升来自类别分布，而 Macro-F1 下降。因此，这组结果只说明两阶段代码链路可以运行，不能作为模型性能提升或论文结果复现的依据。完整记录见 [`RESULTS.md`](experiments/dermamnist_resnet18/RESULTS.md)。

数据集、预训练权重、checkpoint 和论文 PDF 未上传至仓库。

## 论文引用

> Qiu, J., Cao, J., Huang, Y., et al. MUSCLE: A New Perspective to Multi-Scale Fusion for Medical Image Classification Based on the Theory of Evidence. *IEEE Transactions on Medical Imaging*, 45(3), 893–905, 2026. [https://doi.org/10.1109/TMI.2025.3612188](https://doi.org/10.1109/TMI.2025.3612188)

```bibtex
@ARTICLE{11174067,
  author={Qiu, Junlai and Cao, Junyue and Huang, Yawen and Zhu, Ziwei and Wang, Fubo and Lu, Cheng and Li, Yuexiang and Zheng, Yefeng},
  journal={IEEE Transactions on Medical Imaging},
  title={MUSCLE: A New Perspective to Multi-Scale Fusion for Medical Image Classification Based on the Theory of Evidence},
  year={2026},
  volume={45},
  number={3},
  pages={893-905},
  doi={10.1109/TMI.2025.3612188}
}
```
