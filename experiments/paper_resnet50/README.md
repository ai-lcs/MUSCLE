# ResNet-50 三数据集短轮验证

该入口只用于 KvasirV2、ISIC 2018、APTOS 2019 的全量数据链路验证。它依次支持
baseline 与 MUSCLE 两阶段，但不把短轮结果称为论文 Table IV 数值复现。

环境安装分两步，避免从 PyPI 误装 CPU 版 PyTorch：

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r experiments/paper_resnet50/requirements-gpu.txt
```

```powershell
python -m experiments.paper_resnet50 validate-data --dataset kvasirv2 --data-root D:\...\kvasir-dataset-v2
python -m experiments.paper_resnet50 benchmark --dataset kvasirv2 --data-root D:\...\kvasir-dataset-v2
python -m experiments.paper_resnet50 run --dataset kvasirv2 --data-root D:\...\kvasir-dataset-v2 --stage baseline --baseline-epochs 5 --muscle-epochs 3 --resume auto
python -m experiments.paper_resnet50 run --dataset kvasirv2 --data-root D:\...\kvasir-dataset-v2 --stage muscle --baseline-epochs 5 --muscle-epochs 3 --resume auto
```

先对 KvasirV2 benchmark 50 steps。三套合计估时不超过 10 小时才统一用 10+5，
否则统一用 5+3。显存不足时依次使用 `--batch-size 8 --accumulation 2` 和
`--batch-size 4 --accumulation 4`。同一轮三套数据必须使用完全相同的 epoch、batch
与 accumulation 配置。

manifest、checkpoint、权重、完整日志和指标默认写入仓库外的
`D:\杂文件\MUSCLE复现\outputs\paper_resnet50`，均由 `.gitignore` 排除。
