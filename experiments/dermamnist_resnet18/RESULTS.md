## Material Passport

- ID: MUSCLE-DERMAMNIST-1784819788
- Type: Experiment Result
- Status: COMPLETED
- Verification Status: VERIFIED_LIGHTWEIGHT_WORKFLOW
- Run date: 2026-07-23
- Scope: DermaMNIST-64 + ResNet18 代理数据轻量流程

# 3+3 epoch 真实运行结果

## 先给结论

这次实验成功跑通了 MUSCLE 的两阶段代码链路：

```text
预训练 ResNet18
→ baseline 训练并保存 checkpoint
→ 重新加载 checkpoint
→ 冻结骨干网络
→ 从 layer1 到 layer4 提取多尺度特征
→ 产生四份 evidence
→ 训练 MAFC 和证据网络
→ 保存并重新加载 MUSCLE checkpoint
→ 在测试子集上计算指标
```

机制检查全部通过，但模型效果没有得到改善。MUSCLE 的测试 Accuracy 比 baseline 高约 0.021，但 macro-F1 低约 0.052，并且把测试样本全部预测成多数类。因此，这次结果只能证明方法流程能够运行，不能证明 MUSCLE 在该设置下优于 baseline。

## 运行环境与材料

| 项目 | 实际值 |
|---|---|
| Python | 3.11.9 |
| PyTorch | 2.11.0+cpu |
| 设备 | CPU，4 线程 |
| 输入尺寸 | 256×256 |
| batch size | 8 |
| 随机种子 | 1234 |
| baseline epoch | 3 |
| MUSCLE epoch | 3 |
| 优化器 | SGD |
| 初始学习率 | 0.01 |
| momentum | 0.9 |
| weight decay | 0.0001 |
| 学习率策略 | poly |
| 总运行时间 | 121.92 秒 |

数据文件 MD5：

```text
b70a2f5635c6199aeaa28c31d7202e1f
```

ResNet18 官方预训练权重 SHA-256：

```text
f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec
```

## 子集与类别分布

本次采用固定随机种子进行分层抽样，保留原始数据的不平衡趋势。

| 类别编号 | 训练 | 验证 | 测试 |
|---:|---:|---:|---:|
| 0 | 7 | 2 | 5 |
| 1 | 11 | 3 | 7 |
| 2 | 25 | 8 | 15 |
| 3 | 3 | 1 | 2 |
| 4 | 25 | 8 | 15 |
| 5 | 150 | 47 | 94 |
| 6 | 3 | 1 | 2 |
| 合计 | 224 | 70 | 140 |

类别 5 占训练、验证和测试子集约三分之二。类别 3 和类别 6 的测试样本各只有 2 个，因此这组宏平均指标波动很大，不适合做正式性能结论。

## 测试集指标

| 模型 | Accuracy | Macro-Sensitivity | Macro-Specificity | Macro-Precision | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| ResNet18 baseline | 0.6500 | 0.2023 | 0.8930 | 0.1456 | 0.1665 |
| ResNet18 + MUSCLE | 0.6714 | 0.1429 | 0.8571 | 0.0959 | 0.1148 |

MUSCLE 测试集平均 uncertainty 为：

```text
0.5003
```

### Baseline 混淆矩阵

```text
[[0, 0, 1, 0, 0,  4, 0],
 [0, 0, 4, 0, 0,  3, 0],
 [0, 0, 8, 0, 0,  7, 0],
 [0, 0, 1, 0, 0,  1, 0],
 [0, 0, 6, 0, 0,  9, 0],
 [0, 0,11, 0, 0, 83, 0],
 [0, 0, 0, 0, 0,  2, 0]]
```

### MUSCLE 混淆矩阵

```text
[[0, 0, 0, 0, 0,  5, 0],
 [0, 0, 0, 0, 0,  7, 0],
 [0, 0, 0, 0, 0, 15, 0],
 [0, 0, 0, 0, 0,  2, 0],
 [0, 0, 0, 0, 0, 15, 0],
 [0, 0, 0, 0, 0, 94, 0],
 [0, 0, 0, 0, 0,  2, 0]]
```

第二个矩阵清楚显示，MUSCLE 将所有 140 个测试样本都预测为类别 5。Accuracy 看起来略高，只是因为类别 5 本身占多数。

## 机制检查

| 检查项 | 结果 |
|---|---|
| evidence 尺度数 | 4 |
| 每个尺度形状 | `[8, 7]` |
| 聚合 evidence 形状 | `[8, 7]` |
| 最小 evidence | 0.0981，非负 |
| 骨干最大梯度 | 0.0 |
| 融合模块最大梯度 | 0.2495 |
| 训练后骨干参数是否不变 | 是 |
| 训练后融合参数是否改变 | 是 |
| baseline checkpoint 重新加载 | 成功 |
| MUSCLE checkpoint 重新加载 | 成功 |

这些检查支持“代码链路跑通”的结论。

## 与论文设置的关键差异

论文正式实验采用五套论文数据、ResNet50 等骨干、第一阶段 500 epoch、第二阶段 200 epoch和 NVIDIA RTX A6000。本次使用 DermaMNIST、ResNet18、224 个训练样本以及 3+3 epoch。

论文使用 Accuracy、Sensitivity、Specificity、Precision 和 F1-Score。本实验沿用这些指标，但使用宏平均口径来暴露类别不平衡问题。

因此，本结果不能与论文表格进行数值对齐。

## 真实失败记录

第一次单批次检查在加载第二阶段 checkpoint 时失败。

原因不是数据或模型结构，而是原作者通过 `eval()` 拼接 checkpoint 路径。Windows 路径中的 `\v` 被 Python 解释为控制字符，导致文件路径损坏。

新增实验脚本在调用上游模型前将路径转换为正斜杠：

```python
checkpoint_path.resolve().as_posix()
```

修复后，单批次检查和 3+3 epoch 均正常完成。原作者核心文件没有因此被修改。

## 当前结果应该怎样表述

可以说：

> 已在 DermaMNIST-64 和 ResNet18 上跑通 MUSCLE 两阶段轻量方法流程，并验证了 checkpoint、骨干冻结、多尺度 evidence 和融合模块更新。

不能说：

> 已复现 MUSCLE 论文实验结果。

也不能说：

> MUSCLE 在本实验中优于 baseline。

## 下一步

如果继续投入算力，优先级应为：

1. 使用更大训练子集和更长训练；
2. 对类别不平衡采用加权采样或加权损失；
3. 至少运行多个随机种子；
4. 再进行 20+10 epoch 对比；
5. 流程稳定后，再决定是否下载论文中的 KvasirV2。

在这些工作完成前，不建议用当前 Accuracy 作为模型性能证据。
