from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from losses.losses import get_loss
from networks.classification.ResNet import resnet50
from networks.classification.ResNet_Multi_Scale import ResNet_CHW_SAFS_TMSL

from .data import create_datasets


MUSCLE_LAYERS = ["layer1", "layer2", "layer3", "layer4"]
PAPER_TABLE_IV = {
    "isic2018": {
        "baseline": {"ACC": 0.7851, "SEN": 0.6460, "SPE": 0.9496, "PRE": 0.7060, "Macro-F1": 0.6702},
        "muscle": {"ACC": 0.8102, "SEN": 0.6653, "SPE": 0.9530, "PRE": 0.7479, "Macro-F1": 0.7006},
    },
    "aptos2019": {
        "baseline": {"ACC": 0.8076, "SEN": 0.5911, "SPE": 0.9485, "PRE": 0.6702, "Macro-F1": 0.6137},
        "muscle": {"ACC": 0.8076, "SEN": 0.6057, "SPE": 0.9482, "PRE": 0.6763, "Macro-F1": 0.6285},
    },
    "kvasirv2": {
        "baseline": {"ACC": 0.9206, "SEN": 0.9206, "SPE": 0.9887, "PRE": 0.9206, "Macro-F1": 0.9204},
        "muscle": {"ACC": 0.9325, "SEN": 0.9325, "SPE": 0.9904, "PRE": 0.9328, "Macro-F1": 0.9324},
    },
}


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def git_sha(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def environment_info(repo_root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "git_sha": git_sha(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "dependencies": {
            name: _package_version(package)
            for name, package in {
                "albumentations": "albumentations",
                "matplotlib": "matplotlib",
                "numpy": "numpy",
                "opencv": "opencv-python-headless",
                "pandas": "pandas",
                "scikit_learn": "scikit-learn",
            }.items()
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
    }
    if torch.cuda.is_available():
        info.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
        )
    return info


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def checkpoint_path_for_upstream(path: Path) -> str:
    return path.resolve().as_posix()


def build_baseline(num_classes: int, pretrained_path: Path | None = None) -> nn.Module:
    return resnet50(
        weights=True,
        ckpt_path=checkpoint_path_for_upstream(pretrained_path) if pretrained_path else "",
        in_channels=3,
        num_classes=num_classes,
    )


def build_muscle(num_classes: int, baseline_weights: Path) -> ResNet_CHW_SAFS_TMSL:
    model = ResNet_CHW_SAFS_TMSL(
        model_deep=50,
        pretrained=True,
        in_channels=3,
        num_classes=num_classes,
        ckpt_path=checkpoint_path_for_upstream(baseline_weights),
        need_layer_name_list=MUSCLE_LAYERS,
    )
    expected = torch.load(baseline_weights, map_location="cpu", weights_only=True)
    actual = model.original_net.state_dict()
    if expected.keys() != actual.keys() or any(
        not torch.equal(value.cpu(), actual[name].cpu()) for name, value in expected.items()
    ):
        raise ValueError("MUSCLE backbone does not exactly match the baseline checkpoint")
    return model


def create_loaders(
    manifest: dict[str, Any],
    data_root: Path,
    *,
    image_size: int,
    batch_size: int,
    workers: int,
    seed: int,
) -> dict[str, DataLoader]:
    datasets = create_datasets(manifest, data_root, image_size=image_size)
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    return {
        "train": DataLoader(datasets["train"], shuffle=True, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def calculate_metrics(
    targets: Iterable[int], predictions: Iterable[int], class_names: list[str]
) -> dict[str, Any]:
    truth = np.asarray(list(targets), dtype=int)
    predicted = np.asarray(list(predictions), dtype=int)
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(truth, predicted, labels=labels)
    total = int(matrix.sum())
    per_class = []
    for index, name in enumerate(class_names):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        tn = total - tp - fp - fn
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
        per_class.append(
            {
                "class_index": index,
                "class_name": name,
                "support": int(matrix[index, :].sum()),
                "sensitivity": sensitivity,
                "specificity": specificity,
                "precision": precision,
                "f1": f1,
            }
        )
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    predicted_classes = sorted(set(predicted.tolist()))
    missing_predictions = [class_names[i] for i in labels if i not in predicted_classes]
    return {
        "ACC": accuracy,
        "SEN": float(np.mean([row["sensitivity"] for row in per_class])),
        "SPE": float(np.mean([row["specificity"] for row in per_class])),
        "PRE": float(np.mean([row["precision"] for row in per_class])),
        "Macro-F1": float(np.mean([row["f1"] for row in per_class])),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "class_collapse": bool(missing_predictions),
        "unpredicted_classes": missing_predictions,
    }


def save_metric_artifacts(metrics: dict[str, Any], output_dir: Path, stem: str) -> None:
    write_json(output_dir / f"{stem}.json", metrics)
    with (output_dir / f"{stem}_per_class.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics["per_class"][0]))
        writer.writeheader()
        writer.writerows(metrics["per_class"])
    with (output_dir / f"{stem}_confusion.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true\\pred", *[row["class_name"] for row in metrics["per_class"]]])
        for class_row, values in zip(metrics["per_class"], metrics["confusion_matrix"]):
            writer.writerow([class_row["class_name"], *values])
    _save_confusion_png(metrics, output_dir / f"{stem}_confusion.png")


def _save_confusion_png(metrics: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(metrics["confusion_matrix"])
    names = [row["class_name"] for row in metrics["per_class"]]
    figure, axis = plt.subplots(figsize=(max(6, len(names)), max(5, len(names) * 0.8)))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.tick_params(axis="x", rotation=45)
    for row in range(len(names)):
        for column in range(len(names)):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def state_unchanged(before: dict[str, torch.Tensor], module: nn.Module) -> bool:
    after = module.state_dict()
    return before.keys() == after.keys() and all(
        torch.equal(value, after[name].detach().cpu()) for name, value in before.items()
    )


def trainable_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def any_trainable_changed(before: dict[str, torch.Tensor], module: nn.Module) -> bool:
    after = dict(module.named_parameters())
    return any(not torch.equal(value, after[name].detach().cpu()) for name, value in before.items())


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return nullcontext()


def _forward_loss(
    stage: str,
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    epoch: int,
    annealing_step: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
    if stage == "baseline":
        outputs = model(images)
        return criterion(outputs, labels), outputs, None
    evidences, aggregate, _ = model(images)
    loss = get_loss(evidences, aggregate, labels, epoch, aggregate.shape[1], annealing_step, 1, device)
    evidence_info = {
        "view_count": len(evidences),
        "shapes": {str(key): list(value.shape) for key, value in evidences.items()},
        "aggregate_shape": list(aggregate.shape),
        "minimum": min(float(value.min().detach().cpu()) for value in evidences.values()),
    }
    return loss, aggregate, evidence_info


def inspect_muscle_evidence(
    model: nn.Module, loader: DataLoader, device: torch.device, amp: bool
) -> dict[str, Any]:
    model.eval()
    images, _, _ = next(iter(loader))
    images = images.to(device, non_blocking=True)
    with torch.inference_mode(), _autocast(device, amp):
        evidences, aggregate, _ = model(images)
    return {
        "view_count": len(evidences),
        "shapes": {str(key): list(value.shape) for key, value in evidences.items()},
        "aggregate_shape": list(aggregate.shape),
        "minimum": min(float(value.min().cpu()) for value in evidences.values()),
    }


def poly_learning_rate(optimizer: optim.Optimizer, base_lr: float, step: int, total_steps: int) -> float:
    lr = base_lr * (1.0 - min(step / max(total_steps, 1), 1.0)) ** 0.9
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def train_epoch(
    stage: str,
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    amp: bool,
    accumulation: int,
    epoch: int,
    epochs: int,
    base_lr: float,
    annealing_step: int,
    class_names: list[str],
    max_steps: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    set_training_mode(stage, model)
    criterion = nn.CrossEntropyLoss()
    targets: list[int] = []
    predictions: list[int] = []
    losses: list[float] = []
    evidence_info = None
    gpu_health: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    steps = min(len(loader), max_steps) if max_steps else len(loader)
    total_steps = epochs * len(loader)
    started = time.perf_counter()
    for batch_index, (images, labels, _) in enumerate(loader):
        if max_steps is not None and batch_index >= max_steps:
            break
        if device.type == "cuda" and batch_index % 25 == 0:
            health = query_gpu_health()
            gpu_health.append(health)
            if health.get("thermal_slowdown"):
                raise RuntimeError(f"GPU thermal slowdown detected: {health}")
        global_step = (epoch - 1) * len(loader) + batch_index
        lr = poly_learning_rate(optimizer, base_lr, global_step, total_steps)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _autocast(device, amp):
            loss, outputs, current_evidence = _forward_loss(
                stage, model, images, labels, criterion, epoch, annealing_step, device
            )
            scaled_loss = loss / accumulation
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {stage} loss at epoch {epoch}, batch {batch_index}")
        scaler.scale(scaled_loss).backward()
        should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == steps
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if evidence_info is None:
            evidence_info = current_evidence
        losses.append(float(loss.detach().cpu()))
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(outputs.detach().argmax(1).cpu().tolist())
    if device.type == "cuda":
        torch.cuda.synchronize()
    metrics = calculate_metrics(targets, predictions, class_names)
    metrics["loss"] = float(np.mean(losses))
    metrics["learning_rate"] = lr
    metrics["elapsed_seconds"] = time.perf_counter() - started
    metrics["steps"] = len(losses)
    metrics["gpu_health"] = gpu_health
    return metrics, evidence_info


def set_training_mode(stage: str, model: nn.Module) -> None:
    model.train()
    if stage == "muscle":
        model.original_net.eval()


def query_gpu_health() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip()}
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    if len(fields) != 3:
        return {"available": False, "error": result.stdout.strip()}
    software_active = fields[1].casefold() == "active"
    hardware_active = fields[2].casefold() == "active"
    return {
        "available": True,
        "temperature_c": int(fields[0]),
        "software_thermal_slowdown": software_active,
        "hardware_thermal_slowdown": hardware_active,
        "thermal_slowdown": software_active or hardware_active,
    }


def evaluate(
    stage: str,
    model: nn.Module,
    loader: DataLoader,
    *,
    class_names: list[str],
    device: torch.device,
    amp: bool,
    epoch: int,
    annealing_step: int,
) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    uncertainties: list[float] = []
    with torch.inference_mode():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with _autocast(device, amp):
                loss, outputs, _ = _forward_loss(
                    stage, model, images, labels, criterion, epoch, annealing_step, device
                )
            losses.append(float(loss.detach().cpu()))
            targets.extend(labels.cpu().tolist())
            predictions.extend(outputs.argmax(1).cpu().tolist())
            if stage == "muscle":
                uncertainties.extend((len(class_names) / (outputs + 1).sum(1)).cpu().tolist())
    metrics = calculate_metrics(targets, predictions, class_names)
    metrics["loss"] = float(np.mean(losses))
    if uncertainties:
        metrics["mean_uncertainty"] = float(np.mean(uncertainties))
    return metrics


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_acc: float,
    history: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "history": history,
            "config": config,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    expected_config: dict[str, Any] | None = None,
) -> tuple[int, float, list[dict[str, Any]]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if expected_config is not None and checkpoint.get("config") != expected_config:
        raise ValueError("Resume checkpoint configuration differs from the current run")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint.get("scaler", {}))
    random.setstate(checkpoint["rng"]["python"])
    np.random.set_state(checkpoint["rng"]["numpy"])
    torch.set_rng_state(checkpoint["rng"]["torch"])
    if torch.cuda.is_available() and checkpoint["rng"].get("cuda") is not None:
        torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
    return int(checkpoint["epoch"]) + 1, float(checkpoint["best_acc"]), checkpoint["history"]


def save_raw_weights(path: Path, model: nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, path)


def make_conclusion(dataset: str, stage: str, metrics: dict[str, Any]) -> str:
    collapse = "检测到类别坍缩：" + ", ".join(metrics["unpredicted_classes"]) if metrics["class_collapse"] else "未检测到类别坍缩。"
    return (
        f"{dataset} 的 {stage} 短轮验证已完成。测试集 ACC={metrics['ACC']:.4f}，"
        f"Macro-F1={metrics['Macro-F1']:.4f}。{collapse}\n"
        "本结果只验证正式数据与训练链路，不代表论文 Table IV 数值复现。\n"
    )


def table_iv_comparison(dataset: str, stage: str, metrics: dict[str, Any]) -> dict[str, Any]:
    reference = PAPER_TABLE_IV[dataset][stage]
    return {
        "scope": "Descriptive difference only; not an acceptance threshold.",
        "paper_table": "Table IV, ResNet-50 row",
        "paper_reference": reference,
        "short_validation": {name: float(metrics[name]) for name in reference},
        "delta_short_minus_paper": {
            name: float(metrics[name] - reference[name]) for name in reference
        },
    }
