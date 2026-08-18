# 为什么固定聚合值得改：一个从现有仓库出发的设想

> 这一页讲的是研究设想，不是已经完成的改进。本仓库目前已经观察到问题，但还没有实现可靠性折扣，也没有产生改进后的实验结果。

## 先看公开代码做了什么

MUSCLE 从 ResNet-50 的四个 stage 取出特征。每个 stage 经过 MAFC 和 `EvidenceCollector` 后，都会产生一组非负 evidence。evidence 越少，模型计算出的 uncertainty 越高。

公开代码最后使用 `TMSL` 聚合四组 evidence：

```python
evidence_a = evidences[0]
for i in range(1, 4):
    evidence_a = (evidences[i] + evidence_a) / 2
```

这段代码看起来只是不断取平均。把四步展开后，结果却不是四个尺度各占四分之一：

```text
先融合 Stage 1 和 Stage 2：
1/2 × Stage 1 + 1/2 × Stage 2

再加入 Stage 3：
1/4 × Stage 1 + 1/4 × Stage 2 + 1/2 × Stage 3

最后加入 Stage 4：
1/8 × Stage 1 + 1/8 × Stage 2 + 1/4 × Stage 3 + 1/2 × Stage 4
```

因此，Stage 4 固定占最终 evidence 的一半。它权重大，不是因为模型针对当前图像判断它更可信，而是因为它排在最后。如果调换四个尺度的融合顺序，同一组 evidence 也会得到不同结果。

这里并不是说“深层权重大”一定错误。深层特征往往确实更有语义信息。真正值得追问的是：这个权重是否应该永远由排列顺序决定？

## 仓库里的实际观察

本仓库已经复用三套 MUSCLE checkpoint，计算了四个尺度的平均 uncertainty：

| 数据集 | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---:|---:|---:|---:|
| KvasirV2 | 1.0000 | 1.0000 | 1.0000 | 0.0693 |
| ISIC 2018 | 0.6574 | 0.4839 | 0.3640 | 0.2709 |
| APTOS 2019 | 0.4728 | 0.4515 | 0.3674 | 0.2730 |

KvasirV2 最直观：前三个尺度的 uncertainty 接近 `1.0`，说明这些分支在当前短轮 checkpoint 中几乎没有形成有效 evidence；Stage 4 的 uncertainty 只有 `0.0693`。但原来的聚合公式仍然让前三个尺度参与最终结果。

ISIC 2018 和 APTOS 2019 没有这么极端，但也出现了越到深层、平均 uncertainty 越低的趋势。完整结果见[三数据集短轮复现结果](三数据集短轮复现结果.md#四尺度-uncertainty)。

这些数值只能说明“不同尺度的质量确实不一样，值得继续检查”。它们来自 10+5 epoch 的短轮训练，浅层分支不确定性高也可能与训练不足有关，因此不能直接证明原方法错误。

## 改进想法：先判断可信程度，再进行融合

改进不需要再增加一个大型骨干。四个 `EvidenceCollector` 保留不变，只在它们和最终聚合之间增加一步：为每个尺度估计一个 `0–1` 之间的可靠性。

可靠性先看两个容易理解的信号：

- **尺度自身是否有把握**：uncertainty 越高，说明 evidence 越少，可靠性应降低；
- **它和其他尺度是否一致**：如果三个尺度都倾向类别 A，只有一个尺度强烈倾向类别 B，这个分支需要更谨慎地参与融合。

可以先用下面的直观关系理解，而不把它当成已经确定的最终公式：

```text
尺度可靠性 ≈（1 − 自身 uncertainty）× 与其他尺度的一致程度
```

例如，一张图像上 Stage 1 的 uncertainty 是 `0.9`，Stage 4 是 `0.1`，并且 Stage 4 与另外两个尺度判断一致，那么 Stage 4 应保留更多影响，Stage 1 的影响应减小。这里的“折扣”不是武断地删除 Stage 1，而是少相信它对具体类别的支持，并保留更多“尚不确定”的空间。

改进后的区别可以概括为：

| 原始 TMSL | 改进设想 |
|---|---|
| 权重由融合顺序固定 | 权重随当前图像变化 |
| Stage 4 永远占一半 | 哪个尺度可靠就多参考哪个 |
| 已经计算 uncertainty，但不直接用于最终尺度权重 | 用 uncertainty 和尺度间一致性估计可靠性 |

## 先做不重新训练的低成本检查

现有模型已经会返回四组 evidence，因此第一步不需要重新训练 ResNet-50，只需加载已有 checkpoint 重新推理，比较几种聚合方式：

1. 原始固定顺序聚合；
2. 四个尺度等权平均；
3. 只使用 Stage 4；
4. 只使用 Stage 3 和 Stage 4；
5. 调换四个尺度的聚合顺序；
6. 使用 uncertainty 和一致性得到的简单可靠性权重。

这一步主要回答三个问题：改变顺序是否会明显改变结果，去掉高不确定性尺度是否有帮助，简单的自适应权重是否比人工只选深层更好。只有这些推理对照出现稳定信号，才值得进一步微调融合模块。

评价可以先控制在已有的 Accuracy 和 Macro-F1；APTOS 2019 再补充 quadratic weighted Kappa。这样能利用仓库现有数据、checkpoint 和评估代码，不把问题扩大成重新训练多个大型模型。

## 当前仓库已经做到哪一步

| 内容 | 状态 |
|---|---|
| 三套正式数据的 ResNet-50 短轮复现 | 已完成 |
| 四尺度 evidence 和 uncertainty 分析 | 已完成 |
| 固定聚合隐含权重的代码推导 | 已完成 |
| 不同融合顺序的推理对照 | 尚未进行 |
| Stage 3+4、Stage 4 对照 | 尚未进行 |
| 可靠性折扣代码 | 尚未实现 |
| 改进后的性能结果 | 尚不存在 |

因此，目前最准确的表述是：仓库已经提出了一个由代码和复现现象共同引出的可检验问题，下一步可以先用低成本推理判断它是否值得继续，而不能说已经完成了方法改进。

## 与作者后续工作的关系

同一团队的后续工作使用 CNN–ViT 双骨干，并且只融合两个骨干的最后两个 stage。这说明作者也注意到浅层高不确定性信息可能影响融合效果，但他们采用的是人工选择深层特征。

这里保留的研究空间更小，也更适合从现有仓库出发：不增加双骨干，仍使用已经复现的 ResNet-50，让模型针对每张图像决定应该更相信哪些尺度。可靠性折扣本身并不是全新的理论，可能的增量在于用它替代 MUSCLE 的固定顺序和人工尺度选择。

## 参考资料

- Qiu et al., [*MUSCLE: A New Perspective to Multi-Scale Fusion for Medical Image Classification Based on the Theory of Evidence*](https://doi.org/10.1109/TMI.2025.3612188).
- Qiu et al., [*A Hybrid Framework Bridging CNN and ViT based on Theory of Evidence for Diabetic Retinopathy Grading*](https://arxiv.org/abs/2510.26315).
- 本仓库的固定聚合实现：[ResNet `TMSL`](../networks/classification/ResNet_Multi_Scale.py#L97-L112)。

