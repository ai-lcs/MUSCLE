from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.dermamnist_resnet18.dataset import (  # noqa: E402
    DERMAMNIST_CLASSES,
    DermaMNISTNPZ,
)
from losses.losses import get_loss  # noqa: E402
from networks.classification.ResNet import resnet18  # noqa: E402
from networks.classification.ResNet_Multi_Scale import ResNet_CHW_SAFS_TMSL  # noqa: E402


NUM_CLASSES = len(DERMAMNIST_CLASSES)
MUSCLE_LAYERS = ["layer1", "layer2", "layer3", "layer4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight two-stage MUSCLE smoke test on DermaMNIST-64."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--pretrained-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-epochs", type=int, default=3)
    parser.add_argument("--muscle-epochs", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=224)
    parser.add_argument("--val-samples", type=int, default=70)
    parser.add_argument("--test-samples", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--annealing-step", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run one baseline step and one MUSCLE step, then exit.",
    )
    return parser.parse_args()


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.FileHandler(output_dir / "run.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def set_reproducibility(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_loaders(args: argparse.Namespace):
    datasets = {
        "train": DermaMNISTNPZ(
            args.data_path,
            "train",
            max_samples=args.train_samples,
            seed=args.seed,
            image_size=args.image_size,
        ),
        "val": DermaMNISTNPZ(
            args.data_path,
            "val",
            max_samples=args.val_samples,
            seed=args.seed,
            image_size=args.image_size,
        ),
        "test": DermaMNISTNPZ(
            args.data_path,
            "test",
            max_samples=args.test_samples,
            seed=args.seed,
            image_size=args.image_size,
        ),
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }
    return datasets, loaders


def calculate_metrics(targets: list[int], predictions: list[int]) -> dict[str, Any]:
    labels = list(range(NUM_CLASSES))
    matrix = confusion_matrix(targets, predictions, labels=labels)
    specificities = []
    for class_index in labels:
        true_positive = matrix[class_index, class_index]
        false_positive = matrix[:, class_index].sum() - true_positive
        false_negative = matrix[class_index, :].sum() - true_positive
        true_negative = matrix.sum() - true_positive - false_positive - false_negative
        denominator = true_negative + false_positive
        specificities.append(float(true_negative / denominator) if denominator else 0.0)

    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_sensitivity": float(
            recall_score(targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "macro_specificity": float(np.mean(specificities)),
        "macro_precision": float(
            precision_score(targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "confusion_matrix": matrix.tolist(),
    }


def set_poly_learning_rate(
    optimizer: optim.Optimizer,
    base_learning_rate: float,
    global_step: int,
    total_steps: int,
) -> float:
    progress = min(global_step / max(total_steps, 1), 1.0)
    learning_rate = base_learning_rate * (1.0 - progress) ** 0.9
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
    return learning_rate


def evaluate_baseline(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images)
            losses.append(float(criterion(outputs, labels).item()))
            targets.extend(labels.tolist())
            predictions.extend(outputs.argmax(dim=1).tolist())
    metrics = calculate_metrics(targets, predictions)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def evaluate_muscle(
    model: nn.Module,
    loader: DataLoader,
    epoch: int,
    annealing_step: int,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    uncertainties: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            evidences, aggregate_evidence, _ = model(images)
            loss = get_loss(
                evidences,
                aggregate_evidence,
                labels,
                epoch,
                NUM_CLASSES,
                annealing_step,
                1,
                torch.device("cpu"),
            )
            alpha = aggregate_evidence + 1
            uncertainty = NUM_CLASSES / alpha.sum(dim=1)
            losses.append(float(loss.item()))
            uncertainties.extend(uncertainty.tolist())
            targets.extend(labels.tolist())
            predictions.extend(aggregate_evidence.argmax(dim=1).tolist())
    metrics = calculate_metrics(targets, predictions)
    metrics["loss"] = float(np.mean(losses))
    metrics["mean_uncertainty"] = float(np.mean(uncertainties))
    return metrics


def train_baseline(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    *,
    epoch: int,
    total_epochs: int,
    base_learning_rate: float,
) -> dict[str, Any]:
    model.train()
    losses: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    total_steps = total_epochs * len(loader)
    for batch_index, (images, labels) in enumerate(loader):
        global_step = (epoch - 1) * len(loader) + batch_index
        learning_rate = set_poly_learning_rate(
            optimizer, base_learning_rate, global_step, total_steps
        )
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        targets.extend(labels.tolist())
        predictions.extend(outputs.detach().argmax(dim=1).tolist())
    metrics = calculate_metrics(targets, predictions)
    metrics["loss"] = float(np.mean(losses))
    metrics["learning_rate"] = float(learning_rate)
    return metrics


def train_muscle(
    model: ResNet_CHW_SAFS_TMSL,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    *,
    epoch: int,
    total_epochs: int,
    base_learning_rate: float,
    annealing_step: int,
    capture_mechanism: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    model.train()
    model.original_net.eval()
    losses: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    mechanism: dict[str, Any] | None = None
    total_steps = total_epochs * len(loader)

    for batch_index, (images, labels) in enumerate(loader):
        global_step = (epoch - 1) * len(loader) + batch_index
        learning_rate = set_poly_learning_rate(
            optimizer, base_learning_rate, global_step, total_steps
        )
        optimizer.zero_grad(set_to_none=True)
        evidences, aggregate_evidence, _ = model(images)
        loss = get_loss(
            evidences,
            aggregate_evidence,
            labels,
            epoch,
            NUM_CLASSES,
            annealing_step,
            1,
            torch.device("cpu"),
        )
        loss.backward()

        if capture_mechanism and mechanism is None:
            backbone_gradients = [
                parameter.grad.detach().abs().max().item()
                for parameter in model.original_net.parameters()
                if parameter.grad is not None
            ]
            fusion_gradients = [
                parameter.grad.detach().abs().max().item()
                for name, parameter in model.named_parameters()
                if not name.startswith("original_net.")
                and parameter.grad is not None
                and parameter.grad.numel() > 0
            ]
            mechanism = {
                "evidence_view_count": len(evidences),
                "evidence_shapes": {
                    str(index): list(evidence.shape) for index, evidence in evidences.items()
                },
                "aggregate_shape": list(aggregate_evidence.shape),
                "minimum_evidence": float(
                    min(evidence.min().item() for evidence in evidences.values())
                ),
                "backbone_max_gradient": float(max(backbone_gradients, default=0.0)),
                "fusion_max_gradient": float(max(fusion_gradients, default=0.0)),
            }

        optimizer.step()
        losses.append(float(loss.item()))
        targets.extend(labels.tolist())
        predictions.extend(aggregate_evidence.detach().argmax(dim=1).tolist())

    metrics = calculate_metrics(targets, predictions)
    metrics["loss"] = float(np.mean(losses))
    metrics["learning_rate"] = float(learning_rate)
    return metrics, mechanism


def clone_parameters(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in module.named_parameters()}


def parameters_unchanged(
    before: dict[str, torch.Tensor],
    module: nn.Module,
) -> bool:
    current = dict(module.named_parameters())
    return all(torch.equal(value, current[name].detach()) for name, value in before.items())


def any_parameter_changed(
    before: dict[str, torch.Tensor],
    module: nn.Module,
    *,
    excluded_prefix: str,
) -> bool:
    current = dict(module.named_parameters())
    return any(
        not torch.equal(value, current[name].detach())
        for name, value in before.items()
        if not name.startswith(excluded_prefix)
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "num_classes": NUM_CLASSES,
        },
        path,
    )


def build_baseline(args: argparse.Namespace) -> nn.Module:
    return resnet18(
        weights=True,
        ckpt_path=str(args.pretrained_path),
        in_channels=3,
        num_classes=NUM_CLASSES,
    )


def build_muscle(checkpoint_path: Path) -> ResNet_CHW_SAFS_TMSL:
    # The upstream loader builds an eval() string. Forward slashes prevent
    # Windows path fragments such as "\v" from becoming escape characters.
    checkpoint_path_for_upstream = checkpoint_path.resolve().as_posix()
    return ResNet_CHW_SAFS_TMSL(
        model_deep=18,
        pretrained=True,
        in_channels=3,
        num_classes=NUM_CLASSES,
        ckpt_path=checkpoint_path_for_upstream,
        need_layer_name_list=MUSCLE_LAYERS,
    )


def verify_one_batch(
    args: argparse.Namespace,
    train_loader: DataLoader,
) -> dict[str, Any]:
    images, labels = next(iter(train_loader))
    criterion = nn.CrossEntropyLoss()

    baseline = build_baseline(args)
    baseline_optimizer = optim.SGD(
        baseline.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    baseline_optimizer.zero_grad(set_to_none=True)
    baseline_outputs = baseline(images)
    baseline_loss = criterion(baseline_outputs, labels)
    baseline_loss.backward()
    baseline_optimizer.step()

    checkpoint_path = args.output_dir / "checkpoints" / "verify_baseline.pth"
    save_checkpoint(checkpoint_path, baseline, 1, {"loss": float(baseline_loss.item())})

    muscle = build_muscle(checkpoint_path)
    trainable_parameters = [parameter for parameter in muscle.parameters() if parameter.requires_grad]
    muscle_optimizer = optim.SGD(
        trainable_parameters,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    backbone_before = clone_parameters(muscle.original_net)
    all_before = clone_parameters(muscle)
    muscle_metrics, mechanism = train_muscle(
        muscle,
        DataLoader(
            list(zip(images, labels)),
            batch_size=len(labels),
            shuffle=False,
            num_workers=0,
        ),
        muscle_optimizer,
        epoch=1,
        total_epochs=1,
        base_learning_rate=args.learning_rate,
        annealing_step=args.annealing_step,
        capture_mechanism=True,
    )
    assert mechanism is not None
    mechanism.update(
        {
            "baseline_output_shape": list(baseline_outputs.shape),
            "baseline_loss": float(baseline_loss.item()),
            "muscle_loss": muscle_metrics["loss"],
            "backbone_parameters_unchanged": parameters_unchanged(
                backbone_before, muscle.original_net
            ),
            "fusion_parameters_changed": any_parameter_changed(
                all_before,
                muscle,
                excluded_prefix="original_net.",
            ),
        }
    )
    return mechanism


def main() -> int:
    args = parse_args()
    configure_logging(args.output_dir)
    set_reproducibility(args.seed, args.threads)
    started_at = time.time()

    for path in (args.data_path, args.pretrained_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    datasets, loaders = create_loaders(args)
    run_config = {
        "scope": "Lightweight proxy-method smoke test; not a paper-table reproduction.",
        "device": "cpu",
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "data_md5": file_hash(args.data_path, "md5"),
        "pretrained_weight_sha256": file_hash(args.pretrained_path, "sha256"),
        "class_names": DERMAMNIST_CLASSES,
        "split_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "class_counts": {name: dataset.class_counts() for name, dataset in datasets.items()},
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info("Run configuration: %s", json.dumps(run_config, ensure_ascii=False))

    if args.verify_only:
        verification = verify_one_batch(args, loaders["train"])
        verification["elapsed_seconds"] = time.time() - started_at
        expected_output_shape = verification["baseline_output_shape"]
        verification["passed"] = bool(
            verification["baseline_output_shape"][1] == NUM_CLASSES
            and verification["evidence_view_count"] == 4
            and all(
                shape == expected_output_shape
                for shape in verification["evidence_shapes"].values()
            )
            and verification["aggregate_shape"] == expected_output_shape
            and verification["minimum_evidence"] >= 0
            and verification["backbone_max_gradient"] == 0
            and verification["fusion_max_gradient"] > 0
            and verification["backbone_parameters_unchanged"]
            and verification["fusion_parameters_changed"]
        )
        (args.output_dir / "verification.json").write_text(
            json.dumps(verification, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logging.info("Verification result: %s", json.dumps(verification, ensure_ascii=False))
        return 0 if verification["passed"] else 2

    criterion = nn.CrossEntropyLoss()
    baseline = build_baseline(args)
    baseline_optimizer = optim.SGD(
        baseline.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    baseline_checkpoint = args.output_dir / "checkpoints" / "baseline_best.pth"
    baseline_history = []
    best_baseline_f1 = -1.0
    for epoch in range(1, args.baseline_epochs + 1):
        epoch_started = time.time()
        train_metrics = train_baseline(
            baseline,
            loaders["train"],
            baseline_optimizer,
            criterion,
            epoch=epoch,
            total_epochs=args.baseline_epochs,
            base_learning_rate=args.learning_rate,
        )
        val_metrics = evaluate_baseline(baseline, loaders["val"], criterion)
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "elapsed_seconds": time.time() - epoch_started,
        }
        baseline_history.append(record)
        logging.info("Baseline epoch %s: %s", epoch, json.dumps(record, ensure_ascii=False))
        if val_metrics["macro_f1"] > best_baseline_f1:
            best_baseline_f1 = val_metrics["macro_f1"]
            save_checkpoint(baseline_checkpoint, baseline, epoch, val_metrics)

    baseline_checkpoint_data = torch.load(
        baseline_checkpoint, map_location="cpu", weights_only=False
    )
    baseline.load_state_dict(baseline_checkpoint_data["model"])
    baseline_test = evaluate_baseline(baseline, loaders["test"], criterion)

    muscle = build_muscle(baseline_checkpoint)
    trainable_parameters = [parameter for parameter in muscle.parameters() if parameter.requires_grad]
    muscle_optimizer = optim.SGD(
        trainable_parameters,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    backbone_before = clone_parameters(muscle.original_net)
    all_before = clone_parameters(muscle)
    muscle_checkpoint = args.output_dir / "checkpoints" / "muscle_best.pth"
    muscle_history = []
    mechanism: dict[str, Any] | None = None
    best_muscle_f1 = -1.0
    for epoch in range(1, args.muscle_epochs + 1):
        epoch_started = time.time()
        train_metrics, current_mechanism = train_muscle(
            muscle,
            loaders["train"],
            muscle_optimizer,
            epoch=epoch,
            total_epochs=args.muscle_epochs,
            base_learning_rate=args.learning_rate,
            annealing_step=args.annealing_step,
            capture_mechanism=mechanism is None,
        )
        if mechanism is None:
            mechanism = current_mechanism
        val_metrics = evaluate_muscle(
            muscle,
            loaders["val"],
            epoch=epoch,
            annealing_step=args.annealing_step,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "elapsed_seconds": time.time() - epoch_started,
        }
        muscle_history.append(record)
        logging.info("MUSCLE epoch %s: %s", epoch, json.dumps(record, ensure_ascii=False))
        if val_metrics["macro_f1"] > best_muscle_f1:
            best_muscle_f1 = val_metrics["macro_f1"]
            save_checkpoint(muscle_checkpoint, muscle, epoch, val_metrics)

    assert mechanism is not None
    mechanism["backbone_parameters_unchanged"] = parameters_unchanged(
        backbone_before, muscle.original_net
    )
    mechanism["fusion_parameters_changed"] = any_parameter_changed(
        all_before,
        muscle,
        excluded_prefix="original_net.",
    )

    muscle_checkpoint_data = torch.load(
        muscle_checkpoint, map_location="cpu", weights_only=False
    )
    muscle.load_state_dict(muscle_checkpoint_data["model"])
    muscle_test = evaluate_muscle(
        muscle,
        loaders["test"],
        epoch=args.muscle_epochs,
        annealing_step=args.annealing_step,
    )

    result = {
        "scope": run_config["scope"],
        "baseline": {
            "history": baseline_history,
            "best_epoch": int(baseline_checkpoint_data["epoch"]),
            "test": baseline_test,
        },
        "muscle": {
            "history": muscle_history,
            "best_epoch": int(muscle_checkpoint_data["epoch"]),
            "test": muscle_test,
        },
        "mechanism_checks": mechanism,
        "elapsed_seconds": time.time() - started_at,
        "checkpoint_paths": {
            "baseline": str(baseline_checkpoint),
            "muscle": str(muscle_checkpoint),
        },
    }
    (args.output_dir / "experiment_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info("Final result: %s", json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
