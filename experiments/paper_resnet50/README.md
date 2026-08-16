# ResNet-50 三数据集全量短轮复现入口

本目录是当前 fork 的主要实验入口，覆盖 KvasirV2、ISIC 2018 和 APTOS 2019。它复用上游 ResNet 与 MUSCLE 核心实现，新增稳定划分、完整数据审计、Windows/GPU 参数化、两阶段 checkpoint、自动续跑、机制门禁、标准化指标和推理型不确定性分析。

已验证配置为 baseline 10 epoch + MUSCLE 5 epoch。该配置使用三套完整数据，但训练轮数短于论文 500+200 epoch，不能称为 Table IV 完整数值复现。

## 环境

实验实测环境为 Python 3.11.9、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128 和 RTX 5060 Laptop GPU。

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r experiments/paper_resnet50/requirements-gpu.txt
```

## 数据目录

`--data-root` 指向每套数据自身的根目录，不要把数据复制到 Git 仓库。

KvasirV2 根目录下应直接包含八个类别目录；ISIC 2018 根目录应包含官方 Training、Validation 和 Test 的 Input 与 GroundTruth 六个目录；APTOS 2019 根目录应包含 `train.csv` 和 `train_images/`。

## 推荐执行顺序

下面以 KvasirV2 为例。ISIC 和 APTOS 分别把 `--dataset` 改为 `isic2018` 和 `aptos2019`。

### 1. 数据门禁

```powershell
python -m experiments.paper_resnet50 validate-data `
  --dataset kvasirv2 --data-root <DATA_ROOT> --output-root <OUTPUT_ROOT> `
  --batch-size 16 --workers 4 --seed 1234 --amp
```

命令生成稳定 manifest 和 `data_audit.json`，检查文件数、标签、类别分布、全部图像解码、内容重复和跨划分重复。默认不应使用 `--skip-decode-all` 或 `--skip-hash-all`，除非只做临时调试。

### 2. 可选 benchmark

```powershell
python -m experiments.paper_resnet50 benchmark `
  --dataset kvasirv2 --data-root <DATA_ROOT> --output-root <OUTPUT_ROOT> `
  --batch-size 16 --workers 4 --seed 1234 --amp
```

benchmark 只测量短 step 的速度和显存，不产生正式训练结果。

### 3. Baseline

```powershell
python -m experiments.paper_resnet50 run `
  --dataset kvasirv2 --data-root <DATA_ROOT> --output-root <OUTPUT_ROOT> `
  --stage baseline --baseline-epochs 10 --muscle-epochs 5 `
  --batch-size 16 --accumulation 1 --workers 4 `
  --seed 1234 --amp --resume auto
```

第一阶段使用 ImageNet-1K V2 ResNet-50 权重，按验证集 ACC 保存最佳权重。有效的 `last.ckpt` 可由 `--resume auto` 自动续跑。

### 4. MUSCLE

```powershell
python -m experiments.paper_resnet50 run `
  --dataset kvasirv2 --data-root <DATA_ROOT> --output-root <OUTPUT_ROOT> `
  --stage muscle --baseline-epochs 10 --muscle-epochs 5 `
  --batch-size 16 --accumulation 1 --workers 4 `
  --seed 1234 --amp --resume auto
```

第二阶段自动定位同一 profile 下的 baseline 最佳权重。程序检查 checkpoint 是否与 `original_net` 一致，并在训练后验证 backbone 参数、BatchNorm buffers 与融合模块更新。

### 5. 不确定性分析

```powershell
python -m experiments.paper_resnet50 analyze-uncertainty `
  --dataset kvasirv2 --data-root <DATA_ROOT> --output-root <OUTPUT_ROOT> `
  --batch-size 16 --workers 4 --seed 1234 --amp
```

该命令不训练模型。它精确回放干净测试集，计算聚合与四尺度 uncertainty、正确与错误预测差异，以及高斯噪声方差 0、10、100、1000、10000 下的指标和不确定性。

## 输出结构

profile 名 `b10_m5_bs16_a1` 表示 baseline 10 epoch、MUSCLE 5 epoch、batch size 16、梯度累积 1。

```text
<OUTPUT_ROOT>/
└─ <dataset>/
   ├─ manifest.json
   ├─ data_audit.json
   ├─ benchmark/
   └─ b10_m5_bs16_a1/
      ├─ baseline/
      │  ├─ checkpoints/
      │  ├─ run_metadata.json
      │  ├─ epoch_log.jsonl
      │  ├─ test_metrics.json
      │  └─ test_metrics_confusion.png
      └─ muscle/
         ├─ checkpoints/
         ├─ mechanism_checks.json
         ├─ test_metrics.json
         └─ uncertainty_analysis/
```

每次运行记录配置、Git SHA、依赖、GPU、manifest 哈希和输入 checkpoint 哈希。完整产物包含本机路径和大体积权重，只应留在私有工作区。

## 已验证结果与测试

三套数据已统一完成 10+5 epoch，均通过数据门禁、checkpoint 加载、四尺度非负 evidence、backbone 参数和 buffers 冻结、融合模块更新与干净结果回放。汇总指标见 [三数据集短轮复现结果](../../docs/三数据集短轮复现结果.md)。

```powershell
python -B -m unittest discover -s tests -p "test_*.py"
```

测试覆盖稳定划分、指标、类别坍缩、Windows checkpoint 路径、续跑 profile、噪声确定性、uncertainty 公式、冻结参数和 buffers 等关键行为。
