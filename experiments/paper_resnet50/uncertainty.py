from __future__ import annotations

import csv
import hashlib
import json
import math
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from .data import center_crop_square, decode_image, make_transform
from .engine import calculate_metrics, environment_info, file_sha256, write_json


PAPER_NOISE_VARIANCES = (0.0, 10.0, 100.0, 1000.0, 10000.0)
REPLAY_METRICS = ("ACC", "SEN", "SPE", "PRE", "Macro-F1")


def evidence_uncertainty(evidence: torch.Tensor) -> torch.Tensor:
    class_count = evidence.shape[1]
    return class_count / (evidence + 1).sum(dim=1)


def stable_noise_seed(seed: int, relative_path: str, variance: float) -> int:
    payload = f"{seed}\0{relative_path}\0{variance:g}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def apply_gaussian_noise(
    image: np.ndarray, *, variance: float, seed: int, relative_path: str
) -> np.ndarray:
    if variance < 0:
        raise ValueError("Gaussian noise variance cannot be negative")
    if variance == 0:
        return image.copy()
    generator = np.random.default_rng(stable_noise_seed(seed, relative_path, variance))
    noise = generator.normal(0.0, math.sqrt(variance), image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


class NoiseManifestDataset(Dataset):
    def __init__(
        self,
        root: Path,
        entries: Iterable[dict[str, Any]],
        *,
        image_size: int,
        center_crop: bool,
        variance: float,
        seed: int,
    ) -> None:
        self.root = root.resolve()
        self.entries = list(entries)
        self.transform = make_transform(train=False, image_size=image_size)
        self.center_crop = center_crop
        self.variance = variance
        self.seed = seed

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        relative_path = entry["path"]
        path = self.root / Path(PurePosixPath(relative_path))
        image = decode_image(path)
        if image is None:
            raise ValueError(f"Could not decode image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.center_crop:
            image = center_crop_square(image)
        image = apply_gaussian_noise(
            image,
            variance=self.variance,
            seed=self.seed,
            relative_path=relative_path,
        )
        tensor = self.transform(image=image)["image"].float()
        return tensor, torch.tensor(int(entry["label"]), dtype=torch.long), relative_path


def compare_replay_metrics(
    observed: dict[str, Any], expected: dict[str, Any], tolerance: float = 1e-12
) -> dict[str, Any]:
    differences = {name: float(observed[name] - expected[name]) for name in REPLAY_METRICS}
    confusion_match = observed["confusion_matrix"] == expected["confusion_matrix"]
    return {
        "tolerance": tolerance,
        "differences": differences,
        "confusion_matrix_exact_match": confusion_match,
        "passed": confusion_match and all(abs(value) <= tolerance for value in differences.values()),
    }


def summarize_uncertainty_trend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    variances = pd.Series([row["variance"] for row in rows], dtype=float)
    means = pd.Series([row["mean_uncertainty"] for row in rows], dtype=float)
    medians = pd.Series([row["median_uncertainty"] for row in rows], dtype=float)
    mean_rho = float(variances.rank().corr(means.rank()))
    median_rho = float(variances.rank().corr(medians.rank()))
    highest_exceeds_clean = bool(means.iloc[-1] > means.iloc[0] and medians.iloc[-1] > medians.iloc[0])
    return {
        "spearman_variance_vs_mean_uncertainty": mean_rho,
        "spearman_variance_vs_median_uncertainty": median_rho,
        "mean_uncertainty_monotonic_non_decreasing": bool(np.all(np.diff(means) >= 0)),
        "median_uncertainty_monotonic_non_decreasing": bool(np.all(np.diff(medians) >= 0)),
        "highest_noise_exceeds_clean": highest_exceeds_clean,
        "paper_noise_trend_supported": bool(
            highest_exceeds_clean and mean_rho >= 0.9 and median_rho >= 0.9
        ),
    }


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return nullcontext()


def _inference_level(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    class_names: list[str],
    variance: float,
    device: torch.device,
    amp: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    targets: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for images, labels, paths in loader:
            images = images.to(device, non_blocking=True)
            with _autocast(device, amp):
                evidences, aggregate, _ = model(images)
            predictions_batch = aggregate.argmax(1).cpu()
            aggregate_uncertainty = evidence_uncertainty(aggregate).cpu()
            stage_uncertainties = {
                int(stage): evidence_uncertainty(evidence).cpu()
                for stage, evidence in evidences.items()
            }
            for index, relative_path in enumerate(paths):
                label = int(labels[index])
                prediction = int(predictions_batch[index])
                row = {
                    "path": relative_path,
                    "variance": variance,
                    "label": label,
                    "prediction": prediction,
                    "correct": label == prediction,
                    "aggregate_uncertainty": float(aggregate_uncertainty[index]),
                }
                row.update(
                    {
                        f"stage_{stage + 1}_uncertainty": float(values[index])
                        for stage, values in sorted(stage_uncertainties.items())
                    }
                )
                rows.append(row)
                targets.append(label)
                predictions.append(prediction)
    return rows, calculate_metrics(targets, predictions, class_names)


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_uncertainty": float(values.mean()),
        "median_uncertainty": float(np.median(values)),
        "std_uncertainty": float(values.std()),
        "q25_uncertainty": float(np.quantile(values, 0.25)),
        "q75_uncertainty": float(np.quantile(values, 0.75)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_density(series: dict[str, np.ndarray], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 61)
    centers = (bins[:-1] + bins[1:]) / 2
    for label, values in series.items():
        density, _ = np.histogram(values, bins=bins, density=True)
        axis.plot(centers, density, label=label, linewidth=1.8)
    axis.set(xlabel="Uncertainty", ylabel="Density", title=title, xlim=(0, 1))
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_noise_trend(summary: list[dict[str, Any]], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = np.arange(len(summary))
    labels = ["clean" if row["variance"] == 0 else f"$10^{int(math.log10(row['variance']))}$" for row in summary]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(positions, [row["mean_uncertainty"] for row in summary], marker="o", label="Mean")
    axis.plot(positions, [row["median_uncertainty"] for row in summary], marker="s", label="Median")
    axis.set(xticks=positions, xticklabels=labels, xlabel="Gaussian noise variance", ylabel="Uncertainty", title=title)
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_noise_examples(
    root: Path,
    entry: dict[str, Any],
    *,
    center_crop: bool,
    seed: int,
    variances: tuple[float, ...],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    relative_path = entry["path"]
    image = decode_image(root / Path(PurePosixPath(relative_path)))
    if image is None:
        raise ValueError(f"Could not decode example image: {relative_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if center_crop:
        image = center_crop_square(image)
    figure, axes = plt.subplots(1, len(variances), figsize=(3 * len(variances), 3))
    for axis, variance in zip(axes, variances):
        noisy = apply_gaussian_noise(
            image, variance=variance, seed=seed, relative_path=relative_path
        )
        axis.imshow(noisy)
        axis.set_title("Clean" if variance == 0 else f"$\\sigma^2={variance:g}$")
        axis.axis("off")
    figure.suptitle(relative_path)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_uncertainty_analysis(
    *,
    dataset: str,
    manifest: dict[str, Any],
    data_root: Path,
    model: torch.nn.Module,
    checkpoint_path: Path,
    expected_metrics: dict[str, Any],
    output_dir: Path,
    repo_root: Path,
    image_size: int,
    batch_size: int,
    workers: int,
    seed: int,
    amp: bool,
    variances: tuple[float, ...] = PAPER_NOISE_VARIANCES,
) -> dict[str, Any]:
    if variances != PAPER_NOISE_VARIANCES:
        raise ValueError(f"Paper comparison requires variances {PAPER_NOISE_VARIANCES}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = model.to(device).eval()
    center_crop = dataset in {"isic2018", "aptos2019"}
    all_rows: list[dict[str, Any]] = []
    rows_by_variance: dict[float, list[dict[str, Any]]] = {}
    summary: list[dict[str, Any]] = []
    clean_by_path: dict[str, float] = {}

    for variance in variances:
        dataset_view = NoiseManifestDataset(
            data_root,
            manifest["splits"]["test"],
            image_size=image_size,
            center_crop=center_crop,
            variance=variance,
            seed=seed,
        )
        loader = DataLoader(
            dataset_view,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        )
        level_rows, metrics = _inference_level(
            model,
            loader,
            class_names=manifest["class_names"],
            variance=variance,
            device=device,
            amp=amp,
        )
        uncertainties = np.asarray(
            [row["aggregate_uncertainty"] for row in level_rows], dtype=float
        )
        if variance == 0:
            clean_by_path = {
                row["path"]: row["aggregate_uncertainty"] for row in level_rows
            }
        deltas = np.asarray(
            [
                row["aggregate_uncertainty"] - clean_by_path[row["path"]]
                for row in level_rows
            ],
            dtype=float,
        )
        summary_row = {
            "variance": variance,
            "sigma": math.sqrt(variance),
            "samples": len(level_rows),
            **_distribution(uncertainties),
            "mean_delta_vs_clean": float(deltas.mean()),
            "fraction_increased_vs_clean": float(np.mean(deltas > 0)),
            **{name: metrics[name] for name in REPLAY_METRICS},
            "class_collapse": metrics["class_collapse"],
        }
        summary.append(summary_row)
        rows_by_variance[variance] = level_rows
        all_rows.extend(level_rows)
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "variance": variance,
                    "ACC": metrics["ACC"],
                    "Macro-F1": metrics["Macro-F1"],
                    "mean_uncertainty": summary_row["mean_uncertainty"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    clean_rows = rows_by_variance[0.0]
    clean_metrics = calculate_metrics(
        [row["label"] for row in clean_rows],
        [row["prediction"] for row in clean_rows],
        manifest["class_names"],
    )
    replay = compare_replay_metrics(clean_metrics, expected_metrics)
    clean_uncertainty = np.asarray(
        [row["aggregate_uncertainty"] for row in clean_rows], dtype=float
    )
    errors = np.asarray([not row["correct"] for row in clean_rows], dtype=int)
    correct_values = clean_uncertainty[errors == 0]
    error_values = clean_uncertainty[errors == 1]
    correctness = {
        "correct_samples": int(len(correct_values)),
        "incorrect_samples": int(len(error_values)),
        "correct_mean_uncertainty": float(correct_values.mean()),
        "incorrect_mean_uncertainty": float(error_values.mean()),
        "correct_median_uncertainty": float(np.median(correct_values)),
        "incorrect_median_uncertainty": float(np.median(error_values)),
        "error_detection_auroc": float(roc_auc_score(errors, clean_uncertainty)),
    }
    stage_summary = []
    stage_series = {}
    for stage in range(1, 5):
        key = f"stage_{stage}_uncertainty"
        values = np.asarray([row[key] for row in clean_rows], dtype=float)
        stage_summary.append({"stage": f"stage_{stage}", **_distribution(values)})
        stage_series[f"Stage {stage}"] = values
    stage_series["Aggregate"] = clean_uncertainty
    trend = summarize_uncertainty_trend(summary)

    _write_csv(output_dir / "per_sample_uncertainty.csv", all_rows)
    _write_csv(output_dir / "noise_summary.csv", summary)
    _write_csv(output_dir / "stage_summary.csv", stage_summary)
    _plot_density(
        stage_series,
        output_dir / "stage_uncertainty_density.png",
        f"{dataset}: uncertainty by ResNet-50 stage",
    )
    _plot_density(
        {
            "Clean" if variance == 0 else f"Variance {variance:g}": np.asarray(
                [row["aggregate_uncertainty"] for row in rows], dtype=float
            )
            for variance, rows in rows_by_variance.items()
        },
        output_dir / "noise_uncertainty_density.png",
        f"{dataset}: uncertainty under Gaussian noise",
    )
    _plot_noise_trend(
        summary,
        output_dir / "noise_uncertainty_trend.png",
        f"{dataset}: uncertainty trend under Gaussian noise",
    )
    _plot_noise_examples(
        data_root,
        manifest["splits"]["test"][0],
        center_crop=center_crop,
        seed=seed,
        variances=variances,
        path=output_dir / "noise_examples.png",
    )

    result = {
        "scope": "Inference-only short validation of paper Fig. 4 and Fig. 6-7 trends; no retraining.",
        "dataset": dataset,
        "manifest_sha256": manifest["sha256"],
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": file_sha256(checkpoint_path),
        },
        "paper_noise_variances": list(variances),
        "uncertainty_formula": "u = K / sum_k(e_k + 1)",
        "clean_replay": replay,
        "correct_vs_incorrect": correctness,
        "noise_summary": summary,
        "noise_trend": trend,
        "stage_summary": stage_summary,
        "environment": environment_info(repo_root),
        "config": {
            "image_size": image_size,
            "batch_size": batch_size,
            "workers": workers,
            "seed": seed,
            "amp": amp,
        },
    }
    write_json(output_dir / "result.json", result)
    conclusion = (
        f"{dataset} 的推理型不确定性验证已完成。干净测试集回放"
        f"{'通过' if replay['passed'] else '未通过'}；论文噪声趋势"
        f"{'得到支持' if trend['paper_noise_trend_supported'] else '未得到完整支持'}。\n"
        "该结果只复查论文 Fig. 4 与 Fig. 6-7 涉及的现象，不代表完整论文复现。\n"
    )
    (output_dir / "conclusion_zh.txt").write_text(conclusion, encoding="utf-8")
    print(conclusion, flush=True)
    return result
