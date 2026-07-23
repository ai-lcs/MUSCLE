from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


DERMAMNIST_CLASSES = [
    "actinic keratoses and intraepithelial carcinoma",
    "basal cell carcinoma",
    "benign keratosis-like lesions",
    "dermatofibroma",
    "melanoma",
    "melanocytic nevi",
    "vascular lesions",
]


def build_transform(train: bool, image_size: int = 256):
    operations = [transforms.Resize((image_size, image_size), antialias=True)]
    if train:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(operations)


class DermaMNISTNPZ(Dataset):
    """Read one official DermaMNIST split from the downloaded NPZ archive."""

    def __init__(
        self,
        npz_path: str | Path,
        split: str,
        *,
        max_samples: int | None = None,
        seed: int = 1234,
        image_size: int = 256,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        npz_path = Path(npz_path)
        if not npz_path.is_file():
            raise FileNotFoundError(f"DermaMNIST archive not found: {npz_path}")

        with np.load(npz_path) as archive:
            images = archive[f"{split}_images"]
            labels = archive[f"{split}_labels"].reshape(-1).astype(np.int64)

        if max_samples is not None and 0 < max_samples < len(labels):
            all_indices = np.arange(len(labels))
            selected_indices, _ = train_test_split(
                all_indices,
                train_size=max_samples,
                random_state=seed,
                shuffle=True,
                stratify=labels,
            )
            selected_indices = np.sort(selected_indices)
            images = images[selected_indices]
            labels = labels[selected_indices]

        self.images = images
        self.labels = labels
        self.transform = build_transform(train=split == "train", image_size=image_size)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index]).convert("RGB")
        return self.transform(image), int(self.labels[index])

    def class_counts(self) -> dict[int, int]:
        classes, counts = np.unique(self.labels, return_counts=True)
        return {int(label): int(count) for label, count in zip(classes, counts)}
