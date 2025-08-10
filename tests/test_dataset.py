import pathlib
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "train"))
import dataset


def test_dataset_env_path(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    monkeypatch.setattr(dataset, "HAS_TORCHVISION", False)
    loader = dataset.get_dataloader(batch_size=4)
    images, labels = next(iter(loader))
    assert images.shape == (4, 3, 32, 32)
    assert labels.shape == (4,)


def test_dataset_explicit_path(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(dataset, "HAS_TORCHVISION", False)
    with caplog.at_level("INFO"):
        loader = dataset.get_dataloader(batch_size=2, data_path=str(tmp_path))
    assert f"Resolved data_path: {tmp_path}" in caplog.text
    images, labels = next(iter(loader))
    assert images.shape == (2, 3, 32, 32)
    assert labels.shape == (2,)


def test_dataset_fallback_path(monkeypatch, caplog):
    monkeypatch.delenv("DATA_PATH", raising=False)
    monkeypatch.setattr(dataset, "HAS_TORCHVISION", False)
    with caplog.at_level("INFO"):
        _ = dataset.get_dataset()
    assert "Resolved data_path: /tmp/data" in caplog.text


def test_dataset_invalid_path(monkeypatch):
    monkeypatch.setattr(dataset, "HAS_TORCHVISION", True)

    class DummyCIFAR10:
        def __init__(self, root, train, download, transform):
            raise OSError("invalid path")

    monkeypatch.setattr(dataset, "datasets", SimpleNamespace(CIFAR10=DummyCIFAR10))
    monkeypatch.setattr(dataset, "transforms", SimpleNamespace(ToTensor=lambda: None))

    with pytest.raises(OSError):
        dataset.get_dataset(data_path="/invalid/path")
