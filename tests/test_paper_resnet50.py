from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn

from experiments.paper_resnet50.data import (
    _split_content_grouped,
    _split_ordered,
    canonical_manifest_hash,
    decode_image,
    make_transform,
    read_manifest,
    write_manifest,
)
from experiments.paper_resnet50.engine import (
    calculate_metrics,
    checkpoint_path_for_upstream,
    clone_state,
    save_checkpoint,
    load_checkpoint,
    query_gpu_health,
    set_training_mode,
    state_unchanged,
    table_iv_comparison,
)
from experiments.paper_resnet50.__main__ import shared_training_profile
from experiments.paper_resnet50.uncertainty import (
    apply_gaussian_noise,
    compare_replay_metrics,
    evidence_uncertainty,
    summarize_uncertainty_trend,
)


class ManifestTests(unittest.TestCase):
    def test_split_is_stable_and_disjoint(self):
        items = [{"path": f"{index:03}.jpg", "label": 0} for index in range(10)]
        train, val, test = _split_ordered(items)
        self.assertEqual([len(train), len(val), len(test)], [7, 1, 2])
        self.assertEqual(len({item["path"] for split in (train, val, test) for item in split}), 10)

    def test_content_duplicates_stay_in_one_split_with_exact_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = []
            for index in range(10):
                path = root / f"{index:03}.jpg"
                path.write_bytes(b"duplicate" if index in (6, 7) else bytes([index]))
                items.append({"path": path.name, "label": 0})
            train, val, test = _split_content_grouped(items, root)
        self.assertEqual([len(train), len(val), len(test)], [7, 1, 2])
        duplicate_splits = [
            split_index
            for split_index, split in enumerate((train, val, test))
            for item in split
            if item["path"] in {"006.jpg", "007.jpg"}
        ]
        self.assertEqual(len(set(duplicate_splits)), 1)

    def test_manifest_hash_round_trip(self):
        manifest = {
            "schema_version": 1,
            "dataset": "test",
            "class_names": ["a"],
            "split_policy": "test",
            "splits": {"train": [], "val": [], "test": []},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            expected = write_manifest(manifest, path)
            loaded = read_manifest(path)
        self.assertEqual(expected, canonical_manifest_hash(manifest))
        self.assertEqual(loaded["sha256"], expected)

    def test_windows_checkpoint_path_uses_forward_slashes(self):
        path = checkpoint_path_for_upstream(Path("folder") / "best_ACC_weights.pth")
        self.assertNotIn("\\", path)

    def test_image_decode_supports_unicode_windows_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "中文目录" / "图像.jpg"
            path.parent.mkdir()
            ok, encoded = cv2.imencode(".jpg", np.zeros((4, 4, 3), dtype=np.uint8))
            self.assertTrue(ok)
            encoded.tofile(path)
            decoded = decode_image(path)
        self.assertEqual(decoded.shape, (4, 4, 3))

    def test_albumentations_two_transform_returns_expected_tensor(self):
        transformed = make_transform(train=True)(
            image=np.zeros((300, 400, 3), dtype=np.uint8)
        )["image"]
        self.assertEqual(tuple(transformed.shape), (3, 256, 256))

    def test_shared_profile_excludes_dataset_and_stage(self):
        base = dict(
            baseline_epochs=5, muscle_epochs=3, batch_size=8, accumulation=2,
            image_size=256, workers=4, seed=1234, amp=True, learning_rate=0.01,
            momentum=0.9, weight_decay=0.0001, annealing_step=50,
        )
        first = shared_training_profile(Namespace(dataset="kvasirv2", stage="baseline", **base))
        second = shared_training_profile(Namespace(dataset="isic2018", stage="muscle", **base))
        self.assertEqual(first, second)


class MetricTests(unittest.TestCase):
    def test_metrics_and_collapse_are_explicit(self):
        metrics = calculate_metrics([0, 1, 2], [0, 0, 0], ["a", "b", "c"])
        self.assertAlmostEqual(metrics["ACC"], 1 / 3)
        self.assertTrue(metrics["class_collapse"])
        self.assertEqual(metrics["unpredicted_classes"], ["b", "c"])
        self.assertEqual(len(metrics["per_class"]), 3)

    def test_table_iv_comparison_is_a_delta_not_a_gate(self):
        metrics = {"ACC": 0.9, "SEN": 0.8, "SPE": 0.9, "PRE": 0.8, "Macro-F1": 0.8}
        comparison = table_iv_comparison("kvasirv2", "baseline", metrics)
        self.assertIn("Descriptive difference only", comparison["scope"])
        self.assertAlmostEqual(comparison["delta_short_minus_paper"]["ACC"], -0.0206)


class UncertaintyTests(unittest.TestCase):
    def test_zero_noise_is_identity_and_noise_is_deterministic(self):
        image = np.full((8, 8, 3), 128, dtype=np.uint8)
        clean = apply_gaussian_noise(image, variance=0, seed=1234, relative_path="a.png")
        first = apply_gaussian_noise(image, variance=100, seed=1234, relative_path="a.png")
        second = apply_gaussian_noise(image, variance=100, seed=1234, relative_path="a.png")
        self.assertTrue(np.array_equal(clean, image))
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, image))

    def test_evidence_uncertainty_matches_dirichlet_formula(self):
        evidence = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        uncertainty = evidence_uncertainty(evidence)
        self.assertTrue(torch.equal(uncertainty, torch.tensor([1.0, 0.5])))

    def test_replay_comparison_requires_metrics_and_confusion_match(self):
        expected = {
            "ACC": 0.5, "SEN": 0.4, "SPE": 0.8, "PRE": 0.3,
            "Macro-F1": 0.35, "confusion_matrix": [[1, 1], [0, 0]],
        }
        self.assertTrue(compare_replay_metrics(expected, expected)["passed"])
        changed = dict(expected, ACC=0.6)
        self.assertFalse(compare_replay_metrics(changed, expected)["passed"])

    def test_positive_noise_trend_is_summarized_without_claiming_causality(self):
        rows = [
            {"variance": variance, "mean_uncertainty": value, "median_uncertainty": value}
            for variance, value in zip((0, 10, 100, 1000, 10000), (0.1, 0.2, 0.3, 0.4, 0.5))
        ]
        trend = summarize_uncertainty_trend(rows)
        self.assertTrue(trend["paper_noise_trend_supported"])
        self.assertTrue(trend["mean_uncertainty_monotonic_non_decreasing"])


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_contains_resume_state(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.ckpt"
            save_checkpoint(
                path, model=model, optimizer=optimizer, scaler=scaler, epoch=2,
                best_acc=0.5, history=[{"epoch": 2}], config={"stage": "baseline"}
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["epoch"], 2)
        self.assertIn("optimizer", payload)
        self.assertIn("rng", payload)

    def test_checkpoint_restores_epoch_and_rejects_config_drift(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.Linear(2, 2).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.ckpt"
            save_checkpoint(
                path, model=model, optimizer=optimizer, scaler=scaler, epoch=2,
                best_acc=0.5, history=[{"epoch": 2}], config={"stage": "baseline"}
            )
            start, best, history = load_checkpoint(
                path, model, optimizer, scaler, expected_config={"stage": "baseline"}
            )
            with self.assertRaises(ValueError):
                load_checkpoint(path, model, optimizer, scaler, expected_config={"stage": "muscle"})
        self.assertEqual((start, best, history), (3, 0.5, [{"epoch": 2}]))

    def test_buffers_are_included_in_freeze_check(self):
        module = nn.BatchNorm1d(2)
        before = clone_state(module)
        module.running_mean.add_(1)
        self.assertFalse(state_unchanged(before, module))

    def test_muscle_training_mode_keeps_backbone_batchnorm_frozen(self):
        class DummyMuscle(nn.Module):
            def __init__(self):
                super().__init__()
                self.original_net = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
                self.fusion = nn.Linear(2, 2)

        model = DummyMuscle()
        set_training_mode("muscle", model)
        self.assertTrue(model.fusion.training)
        self.assertFalse(model.original_net.training)

    def test_gpu_health_schema_when_nvidia_smi_is_available(self):
        health = query_gpu_health()
        if health["available"]:
            self.assertIsInstance(health["temperature_c"], int)
            self.assertIn("thermal_slowdown", health)


if __name__ == "__main__":
    unittest.main()
