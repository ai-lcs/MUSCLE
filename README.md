# MUSCLE 中文学习与轻量复现记录

这是我用于学习论文 **MUSCLE: A New Perspective to Multi-Scale Fusion for Medical Image Classification Based on the Theory of Evidence** 的个人 fork。

MUSCLE 面向医学图像分类。医学图像中的各种乱七八糟的信息可能大小不同、分布零散，网络只使用最后一层特征时，可能遗漏浅层保留的纹理和局部信息。

论文采用的主线可以压缩成：

```text
先训练普通骨干网络
→ 冻结骨干网络
→ 从多个阶段提取不同尺度的特征
→ 使用 MAFC 从通道、高度、宽度三个方向压缩特征
→ 每个尺度产生分类证据和不确定性
→ 聚合多个尺度的意见
→ 得到最终分类结果
```

MUSCLE 支持三类骨干网络：

- ResNet
- VanillaNet
- Swin Transformer

已完成的轻量实验入口：

- [DermaMNIST-64 + ResNet18 轻量流程验证](experiments/dermamnist_resnet18/README.md)

当前尚未完成：

- 完整 DermaMNIST 训练；
- 20+10 epoch 短训练；
- 论文原始数据集和表格指标复现。

```text
MUSCLE/
├─ datasets/                    五套论文数据集的读取与增强
├─ losses/                      证据损失与一致性损失
├─ networks/
│  ├─ classification/          三类骨干网络及其 MUSCLE 版本
│  └─ net_factory_cls.py       根据参数选择模型
├─ main.py                     训练、验证、测试总入口
├─ utils.py                    指标计算与结果保存
├─ docs/                       本 fork 新增的中文学习文档
├─ LICENSE                     原作者 MIT 许可证
└─ README.md                   当前中文入口
```

## 论文使用的数据集

- [ISIC 2018](https://challenge.isic-archive.com/landing/2018/)
- [APTOS 2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
- [KvasirV2](https://datasets.simula.no/kvasir)
- [Chaoyang](https://bupt-ai-cz.github.io/HSA-NRL)
- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert)

数据集体积较大，而且可能有各自的使用条件。本仓库不上传这些数据。

## 复现层级

| 层级 | 可以证明什么 |
|---|---|
| 环境检查 | 依赖、轻量数据和普通模型前向计算可用 |
| 轻量 smoke test | 两阶段训练、checkpoint、骨干冻结和证据支路能够运行 |
| 代理数据短训练 | 在小型代理数据上观察 baseline 与 MUSCLE 的趋势 |
| 论文数据正式实验 | 在对齐数据、模型和训练设置后比较结果 | 

## 原论文与引用

论文信息：

> Qiu, J., Cao, J., Huang, Y., et al. MUSCLE: A New Perspective to Multi-Scale Fusion for Medical Image Classification Based on the Theory of Evidence. IEEE Transactions on Medical Imaging, 45(3), 893-905, 2026.
> DOI: [10.1109/TMI.2025.3612188](https://doi.org/10.1109/TMI.2025.3612188)

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
