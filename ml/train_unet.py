#!/usr/bin/env python3
"""Train the damage U-Net on paired images and binary masks.

Expected layout (filenames must match):

    data/masks/images/*.png
    data/masks/masks/*.png

xBD building chips do not include pixel damage masks. Use this script only
after you create masks. The running API does not need this checkpoint: it
falls back to pre/post change or texture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from unet import build_unet


class MaskDataset(Dataset):
    def __init__(self, root: Path, size: int = 256) -> None:
        self.images = sorted((root / "images").glob("*"))
        self.mask_dir = root / "masks"
        self.size = size
        self.image_tf = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        path = self.images[index]
        image = self.image_tf(Image.open(path).convert("RGB"))
        mask_path = self.mask_dir / path.name
        mask = Image.open(mask_path).convert("L").resize((self.size, self.size))
        mask_t = transforms.ToTensor()(mask)
        mask_t = (mask_t > 0.5).float()
        return image, mask_t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("../data/masks"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("../models/unet_damage.pt"))
    args = parser.parse_args()
    if not (args.data / "images").is_dir() or not (args.data / "masks").is_dir():
        raise SystemExit(f"Need {args.data}/images and {args.data}/masks with matching filenames.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MaskDataset(args.data)
    if len(dataset) == 0:
        raise SystemExit("No mask images found.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = build_unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), masks)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        print(f"epoch {epoch + 1} loss {total / max(len(loader), 1):.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
