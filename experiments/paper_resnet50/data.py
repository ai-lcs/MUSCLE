from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


DATASET_CLASSES = {
    "kvasirv2": (
        "esophagitis",
        "dyed-lifted-polyps",
        "dyed-resection-margins",
        "normal-cecum",
        "normal-pylorus",
        "normal-z-line",
        "polyps",
        "ulcerative-colitis",
    ),
    "isic2018": ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"),
    "aptos2019": ("0", "1", "2", "3", "4"),
}

ISIC_SPLITS = {
    "train": (
        "ISIC2018_Task3_Training_Input",
        "ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv",
    ),
    "val": (
        "ISIC2018_Task3_Validation_Input",
        "ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv",
    ),
    "test": (
        "ISIC2018_Task3_Test_Input",
        "ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv",
    ),
}

EXPECTED_COUNTS = {
    "kvasirv2": {"train": 5600, "val": 800, "test": 1600},
    "isic2018": {"train": 10015, "val": 193, "test": 1512},
    "aptos2019": {"all": 3662},
}

DATASET_PROVENANCE = {
    "kvasirv2": {
        "source": "https://datasets.simula.no/kvasir/",
        "archive": "https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip",
        "usage_note": "Research and educational use; retain the dataset citation.",
    },
    "isic2018": {
        "source": "https://challenge.isic-archive.com/data/",
        "license": "CC-BY-NC",
        "archives": [
            "ISIC2018_Task3_Training_Input.zip",
            "ISIC2018_Task3_Training_GroundTruth.zip",
            "ISIC2018_Task3_Validation_Input.zip",
            "ISIC2018_Task3_Validation_GroundTruth.zip",
            "ISIC2018_Task3_Test_Input.zip",
            "ISIC2018_Task3_Test_GroundTruth.zip",
        ],
    },
    "aptos2019": {
        "source": "https://www.kaggle.com/competitions/aptos2019-blindness-detection/data",
        "access_note": "Kaggle sign-in and acceptance of the competition rules are required.",
    },
}


def _entry(relative_path: Path | PurePosixPath | str, label: int) -> dict[str, Any]:
    return {"path": PurePosixPath(relative_path).as_posix(), "label": int(label)}


def _split_ordered(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], ...]:
    train_end = round(len(items) * 0.7)
    val_end = train_end + round(len(items) * 0.1)
    return items[:train_end], items[train_end:val_end], items[val_end:]


def _split_content_grouped(
    items: list[dict[str, Any]],
    root: Path,
    assigned_hash_splits: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], ...]:
    """Split ordered items without placing identical file content across splits."""
    if assigned_hash_splits is None:
        assigned_hash_splits = {}
    split_names = ("train", "val", "test")
    target_sizes = dict(zip(split_names, (len(part) for part in _split_ordered(items))))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        digest = _file_sha256(root / Path(item["path"]))
        grouped.setdefault(digest, []).append(item)

    result = {split: [] for split in split_names}
    pending: list[tuple[str, list[dict[str, Any]]]] = []
    for digest, group in grouped.items():
        locked_split = assigned_hash_splits.get(digest)
        if locked_split is None:
            pending.append((digest, group))
            continue
        if len(result[locked_split]) + len(group) > target_sizes[locked_split]:
            raise ValueError(
                f"Cannot keep duplicate content in {locked_split} while preserving split sizes"
            )
        result[locked_split].extend(group)

    for digest, group in pending:
        destination = next(
            (
                split
                for split in split_names
                if len(result[split]) + len(group) <= target_sizes[split]
            ),
            None,
        )
        if destination is None:
            raise ValueError("Cannot preserve split sizes while grouping duplicate content")
        result[destination].extend(group)
        assigned_hash_splits[digest] = destination

    if any(len(result[split]) != target_sizes[split] for split in split_names):
        raise ValueError("Content-grouped split did not reach the requested split sizes")
    return tuple(result[split] for split in split_names)


def build_kvasir_manifest(root: Path) -> dict[str, list[dict[str, Any]]]:
    result = {"train": [], "val": [], "test": []}
    assigned_hash_splits: dict[str, str] = {}
    for label, class_name in enumerate(DATASET_CLASSES["kvasirv2"]):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(class_dir)
        items = [
            _entry(path.relative_to(root), label)
            for path in sorted(
                class_dir.iterdir(), key=lambda item: (item.name.casefold(), item.name)
            )
            if path.is_file()
        ]
        if len(items) != 1000:
            raise ValueError(f"KvasirV2 class {class_name} expected 1000 images, found {len(items)}")
        train, val, test = _split_content_grouped(items, root, assigned_hash_splits)
        result["train"].extend(train)
        result["val"].extend(val)
        result["test"].extend(test)
    return result


def _isic_label(row: pd.Series, class_names: tuple[str, ...]) -> int:
    if "argmax" in row.index:
        return int(row["argmax"])
    values = [float(row[name]) for name in class_names]
    positive = [index for index, value in enumerate(values) if value == 1.0]
    if len(positive) != 1:
        raise ValueError(f"ISIC row must contain exactly one positive label, got {values}")
    return positive[0]


def build_isic_manifest(root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    class_names = DATASET_CLASSES["isic2018"]
    for split, (image_dir_name, csv_name) in ISIC_SPLITS.items():
        image_dir = root / image_dir_name
        csv_path = root / Path(csv_name)
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        table = pd.read_csv(csv_path)
        if "image" not in table.columns:
            raise ValueError(f"ISIC label file has no image column: {csv_path}")
        entries = []
        for _, row in table.iterrows():
            image_path = image_dir / f"{row['image']}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            entries.append(_entry(image_path.relative_to(root), _isic_label(row, class_names)))
        expected = EXPECTED_COUNTS["isic2018"][split]
        if len(entries) != expected:
            raise ValueError(f"ISIC {split} expected {expected} rows, found {len(entries)}")
        result[split] = entries
    return result


def build_aptos_manifest(root: Path) -> dict[str, list[dict[str, Any]]]:
    csv_path = root / "train.csv"
    image_dir = root / "train_images"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    table = pd.read_csv(csv_path)
    required = {"id_code", "diagnosis"}
    if not required.issubset(table.columns):
        raise ValueError(f"APTOS label file must contain {sorted(required)}")
    if len(table) != EXPECTED_COUNTS["aptos2019"]["all"]:
        raise ValueError(f"APTOS expected 3662 rows, found {len(table)}")

    result = {"train": [], "val": [], "test": []}
    assigned_hash_splits: dict[str, str] = {}
    for label in range(5):
        rows = table[table["diagnosis"] == label]
        items = []
        for _, row in rows.iterrows():
            image_path = image_dir / f"{row['id_code']}.png"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            items.append(_entry(image_path.relative_to(root), label))
        train, val, test = _split_content_grouped(items, root, assigned_hash_splits)
        result["train"].extend(train)
        result["val"].extend(val)
        result["test"].extend(test)
    return result


def build_manifest(dataset_name: str, root: Path) -> dict[str, Any]:
    root = root.resolve()
    builders = {
        "kvasirv2": build_kvasir_manifest,
        "isic2018": build_isic_manifest,
        "aptos2019": build_aptos_manifest,
    }
    if dataset_name not in builders:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    splits = builders[dataset_name](root)
    expected_labels = set(range(len(DATASET_CLASSES[dataset_name])))
    actual_labels = {
        int(entry["label"])
        for entries in splits.values()
        for entry in entries
    }
    total_entries = sum(len(entries) for entries in splits.values())
    if actual_labels != expected_labels:
        raise ValueError(
            f"{dataset_name} labels differ from expected {sorted(expected_labels)}: "
            f"found {sorted(actual_labels)}"
        )
    if dataset_name == "aptos2019" and total_entries != EXPECTED_COUNTS[dataset_name]["all"]:
        raise ValueError(f"APTOS split lost rows: expected 3662, found {total_entries}")
    return {
        "schema_version": 1,
        "dataset": dataset_name,
        "class_names": list(DATASET_CLASSES[dataset_name]),
        "provenance": DATASET_PROVENANCE[dataset_name],
        "split_policy": {
            "kvasirv2": (
                "sorted filenames per class, exact-content groups kept together, 70/10/20"
            ),
            "isic2018": "official training/validation/test split",
            "aptos2019": (
                "source CSV order per class, exact-content groups kept together, 70/10/20"
            ),
        }[dataset_name],
        "splits": splits,
    }


def canonical_manifest_hash(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_manifest(manifest: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_hash = canonical_manifest_hash(manifest)
    payload = dict(manifest)
    payload["sha256"] = manifest_hash
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_hash


def read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = manifest.pop("sha256", None)
    actual_hash = canonical_manifest_hash(manifest)
    if stored_hash != actual_hash:
        raise ValueError(f"Manifest hash mismatch: {path}")
    manifest["sha256"] = actual_hash
    return manifest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_image(path: Path) -> Any:
    # cv2.imread cannot reliably open Unicode paths on Windows.
    encoded = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def audit_manifest(
    manifest: dict[str, Any],
    root: Path,
    *,
    decode_all: bool = True,
    hash_all: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    seen_paths: set[str] = set()
    file_hashes: dict[str, tuple[str, str]] = {}
    split_reports = {}
    corrupt_files = []
    duplicate_content = []
    for split in ("train", "val", "test"):
        entries = manifest["splits"][split]
        counts = Counter(int(entry["label"]) for entry in entries)
        for entry in entries:
            relative = entry["path"]
            if relative in seen_paths:
                raise ValueError(f"Path appears in more than one split: {relative}")
            seen_paths.add(relative)
            path = root / Path(PurePosixPath(relative))
            if not path.is_file():
                raise FileNotFoundError(path)
            if decode_all:
                image = decode_image(path)
                if image is None or image.size == 0:
                    corrupt_files.append(relative)
            if hash_all:
                digest = _file_sha256(path)
                previous = file_hashes.get(digest)
                if previous is not None:
                    previous_split, previous_path = previous
                    duplicate_content.append(
                        {
                            "first": previous_path,
                            "first_split": previous_split,
                            "second": relative,
                            "second_split": split,
                            "cross_split": previous_split != split,
                        }
                    )
                else:
                    file_hashes[digest] = (split, relative)
        split_reports[split] = {
            "samples": len(entries),
            "class_counts": {str(label): counts.get(label, 0) for label in range(len(manifest["class_names"]))},
        }
    cross_split_duplicates = [item for item in duplicate_content if item["cross_split"]]
    return {
        "dataset": manifest["dataset"],
        "manifest_sha256": manifest.get("sha256") or canonical_manifest_hash(manifest),
        "splits": split_reports,
        "corrupt_files": corrupt_files,
        "duplicate_content": duplicate_content,
        "cross_split_duplicates": cross_split_duplicates,
        "passed": not corrupt_files and not cross_split_duplicates,
    }


def make_transform(train: bool, image_size: int = 256) -> A.Compose:
    transforms: list[Any] = [A.Resize(image_size, image_size)]
    if train:
        transforms.extend(
            [
                A.OneOf(
                    [
                        A.HorizontalFlip(p=1),
                        A.VerticalFlip(p=1),
                        A.RandomRotate90(p=1),
                        A.Transpose(p=1),
                        A.RandomResizedCrop(
                            size=(image_size, image_size),
                            scale=(0.8, 1.0),
                            ratio=(0.8, 1.0),
                            p=1,
                        ),
                    ],
                    p=0.8,
                ),
                A.OneOf(
                    [
                        A.ColorJitter(
                            brightness=0.2, contrast=0.1, saturation=0.1, hue=0.05, p=1
                        ),
                        A.Blur(blur_limit=5, p=1),
                        A.GaussNoise(p=1),
                        A.ISONoise(p=1),
                        A.GaussianBlur(blur_limit=(3, 5), p=1),
                        A.CoarseDropout(
                            num_holes_range=(1, 5),
                            hole_height_range=(5, 20),
                            hole_width_range=(5, 20),
                            p=1,
                        ),
                    ],
                    p=0.8,
                ),
            ]
        )
    transforms.append(ToTensorV2())
    return A.Compose(transforms)


def center_crop_square(image: Any) -> Any:
    height, width = image.shape[:2]
    edge = min(height, width)
    top = (height - edge) // 2
    left = (width - edge) // 2
    return image[top : top + edge, left : left + edge]


class ManifestDataset(Dataset):
    def __init__(
        self,
        root: Path,
        entries: Iterable[dict[str, Any]],
        *,
        train: bool,
        image_size: int,
        center_crop: bool,
    ) -> None:
        self.root = root.resolve()
        self.entries = list(entries)
        self.transform = make_transform(train=train, image_size=image_size)
        self.center_crop = center_crop

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        path = self.root / Path(PurePosixPath(entry["path"]))
        image = decode_image(path)
        if image is None:
            raise ValueError(f"Could not decode image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.center_crop:
            image = center_crop_square(image)
        tensor = self.transform(image=image)["image"].float()
        return tensor, torch.tensor(int(entry["label"]), dtype=torch.long), entry["path"]


def create_datasets(
    manifest: dict[str, Any], root: Path, *, image_size: int
) -> dict[str, ManifestDataset]:
    center_crop = manifest["dataset"] in {"isic2018", "aptos2019"}
    return {
        split: ManifestDataset(
            root,
            manifest["splits"][split],
            train=split == "train",
            image_size=image_size,
            center_crop=center_crop,
        )
        for split in ("train", "val", "test")
    }
