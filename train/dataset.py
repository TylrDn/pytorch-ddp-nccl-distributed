from __future__ import annotations

import logging
import os

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


def get_dataset(data_path: str | None = None) -> Dataset:
    """Return CIFAR10 or a synthetic dataset.

    Parameters
    ----------
    data_path:
        Directory to download/load the dataset from. If ``None`` the path is
        read from the ``DATA_PATH`` environment variable and defaults to
        ``/tmp/data`` when unset.
    """

    if data_path is None:
        data_path = os.environ.get("DATA_PATH", "/tmp/data")
        logging.info("Using data_path: %s", data_path)
    else:
        logging.debug("Using data_path: %s", data_path)

    if HAS_TORCHVISION:
        transform = transforms.ToTensor()
        return datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    return RandomDataset()


def get_dataloader(batch_size: int = 4, data_path: str | None = None) -> DataLoader:
    """Return a dataloader for the configured dataset."""

    return DataLoader(get_dataset(data_path), batch_size=batch_size)
