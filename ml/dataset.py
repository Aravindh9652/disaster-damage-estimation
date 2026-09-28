from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

try:  # Package import for the API; direct import for existing training scripts.
    from .labels import CLASS_TO_IDX
except ImportError:
    from labels import CLASS_TO_IDX


def train_transforms(size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.15, 0.15, 0.1, 0.05),
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def eval_transforms(size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class BuildingCropDataset(Dataset):
    def __init__(self, csv_path: Path, root: Path, transform=None) -> None:
        self.root = root
        self.transform = transform
        with csv_path.open() as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        path = self.root / row["path"]
        if not path.exists():
            path = Path(row["path"])
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = CLASS_TO_IDX[row["label"]]
        return image, label, row["uid"]
