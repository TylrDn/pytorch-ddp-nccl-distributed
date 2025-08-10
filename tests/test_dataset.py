import pathlib
import sys

import pytest

pytest.importorskip("torch")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "train"))
import dataset


def test_dataset_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    loader = dataset.get_dataloader(batch_size=4)
    images, labels = next(iter(loader))
    assert images.shape == (4, 3, 32, 32)
    assert labels.shape == (4,)
