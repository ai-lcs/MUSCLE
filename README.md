# MUSCLE 中文学习与轻量复现记录

这是我用于学习论文 **MUSCLE: A New Perspective to Multi-Scale Fusion for Medical Image Classification Based on the Theory of Evidence** 的个人 fork。

原始代码来自 [Q4CS/MUSCLE](https://github.com/Q4CS/MUSCLE)。本仓库保留作者公开的模型、数据读取和训练代码，并增加面向初学者的中文说明。这里不是官方中文仓库，也不代表原作者对中文内容进行了审核。

## 这个项目研究什么

MUSCLE 面向医学图像分类。医学图像中的病灶可能大小不同、分布零散，网络只使用最后一层特征时，可能遗漏浅层保留的纹理和局部信息。

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

## 当前仓库处于什么状态

原作者的论文实现源码没有因为中文化而重命名类、函数、参数或目录。本 fork 另外增加了独立的 `experiments/` 目录，用于教学型轻量实验。

中文整理完成的内容包括：

- 仓库入口与风险说明；
- 面向初学者的阅读顺序；
- 代码目录和训练调用链；
- 论文方法与代码文件的对应关系；
- 轻量复现与论文结果复现的边界；
- 哪些文件可以上传 GitHub，哪些必须只留在本机；
- DermaMNIST-64 + ResNet18 的单批次机制检查；
- baseline 3 epoch + MUSCLE 3 epoch 的两阶段轻量运行。

已完成的轻量实验入口：

- [DermaMNIST-64 + ResNet18 轻量流程验证](experiments/dermamnist_resnet18/README.md)

当前尚未完成：

- 完整 DermaMNIST 训练；
- 20+10 epoch 短训练；
- 论文原始数据集和表格指标复现。

因此，现在可以说“已经在代理数据和较小骨干上跑通两阶段轻量流程”，不能说“已经复现论文结果”。

## 初学者从哪里开始

建议按下面的顺序阅读：

1. [从这里开始：先建立项目地图](docs/01-从这里开始.md)
2. [代码结构导读：每个文件夹负责什么](docs/02-代码结构导读.md)
3. [论文方法与代码怎样对应](docs/03-论文方法与代码对应.md)
4. [轻量复现边界与后续计划](docs/04-轻量复现边界与计划.md)
5. [哪些文件应保留，哪些不能上传](docs/05-文件保留与上传规则.md)
6. [查看已经实际运行的轻量实验](experiments/dermamnist_resnet18/README.md)

如果还没有学习 Python、PyTorch 和深度学习，不建议从 `main.py` 第一行开始逐行阅读。先看中文导读，再回到具体文件，会更容易判断自己卡在 Python、PyTorch、模型原理还是运行环境。

## 原始代码目录

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

## 运行前必须知道的限制

原始代码更接近“论文实现代码”，并不是开箱即用的软件：

- 多个数据读取文件把路径写成了 `/home/datasets/...`，不能直接在普通 Windows 电脑上运行；
- `main.py` 默认使用 CheXpert、MUSCLE 模型、200 个 epoch 和 batch size 16；
- 训练 DataLoader 默认设置 `num_workers=32`，Windows 入门环境通常需要先调低；
- MUSCLE 是两阶段训练，第二阶段需要第一阶段训练得到的 checkpoint；
- 官方仓库没有提供论文实验对应的第一阶段 checkpoint；
- 完整数据、模型权重和训练输出没有包含在 GitHub 仓库中。

请不要在没有修改路径和确认硬件环境的情况下直接运行默认 `main.py`。

## 官方依赖

原作者 README 列出的主要版本为：

```text
Python 3.10
PyTorch 2.0.1 + CUDA 11.7
NumPy 1.26.4
Pillow 10.4.0
OpenCV-Python 4.10.0.84
Pandas 2.2.2
```

这些版本来自论文代码发布环境，不代表它们适合所有新显卡。尤其是较新的 NVIDIA 显卡，不应只为了照抄论文版本而强行安装过旧的 CUDA 组合。

## 论文使用的数据集

- [ISIC 2018](https://challenge.isic-archive.com/landing/2018/)
- [APTOS 2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
- [KvasirV2](https://datasets.simula.no/kvasir)
- [Chaoyang](https://bupt-ai-cz.github.io/HSA-NRL)
- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert)

数据集体积较大，而且可能有各自的使用条件。本仓库不上传这些数据。

## 复现层级

| 层级 | 可以证明什么 | 不能声称什么 |
|---|---|---|
| 环境检查 | 依赖、轻量数据和普通模型前向计算可用 | MUSCLE 已经复现 |
| 轻量 smoke test | 两阶段训练、checkpoint、骨干冻结和证据支路能够运行 | 得到论文指标 |
| 代理数据短训练 | 在小型代理数据上观察 baseline 与 MUSCLE 的趋势 | 复现论文表格 |
| 论文数据正式实验 | 在对齐数据、模型和训练设置后比较结果 | 未对齐条件下声称完整复现 |

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

## 许可证与署名

原始代码采用 [MIT License](LICENSE)，版权归 Q4CS 所有。MIT 许可证是法律文本，因此保留英文原文，不做中文替换。

本 fork 新增的中文文档用于个人学习与研究记录。引用或使用原代码时，请保留原作者版权信息并引用论文。
