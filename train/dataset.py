from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

try:  # optional dependency
    from torchvision import datasets, transforms

    HAS_TORCHVISION = True
except Exception:  # pragma: no cover - handled when torchvision missing
    HAS_TORCHVISION = False


class RandomDataset(Dataset):
    def __init__(self, length: int = 1024, num_classes: int = 10) -> None:
        self.length = length
        self.num_classes = num_classes

    def __len__(self) -> int:  # noqa: D401 - simple return
        return self.length

    def __getitem__(self, index: int):  # noqa: D401 - simple return
        x = torch.randn(3, 32, 32)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y


def get_dataset() -> Dataset:
    if HAS_TORCHVISION:
        transform = transforms.ToTensor()
        return datasets.CIFAR10(root="/tmp/data", train=True, download=True, transform=transform)
    return RandomDataset()


def get_dataloader(batch_size: int = 4) -> DataLoader:
    return DataLoader(get_dataset(), batch_size=batch_size)
