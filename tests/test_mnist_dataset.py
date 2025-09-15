import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ddpm_model.datasets.MnistDatasets import MNISTDataset


def _write_img(path: Path, value: int, size=(4, 4)):
    arr = np.full((size[1], size[0]), fill_value=value, dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    img.save(path)


def test_mnist_dataset_missing_root_raises(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        _ = MNISTDataset(dataset_split="train", data_root=str(missing))


def test_mnist_dataset_loads_and_normalizes(tmp_path: Path):
    # Create directories for labels 0 and 1
    root = tmp_path
    (root / "0").mkdir(parents=True)
    (root / "1").mkdir(parents=True)

    # Write simple images: 0 -> black (0), 1 -> white (255)
    _write_img(root / "0" / "a.png", 0)
    _write_img(root / "1" / "b.png", 255)

    ds = MNISTDataset(dataset_split="train", data_root=str(root))

    assert len(ds) == 2
    # Check labels metadata collected
    assert sorted(ds.labels) == [0, 1]

    # Fetch tensors and validate normalization to [-1, 1]
    # Do not assume ordering from filesystem; index by label metadata
    idx0 = ds.labels.index(0)
    idx1 = ds.labels.index(1)
    x0 = ds[idx0]
    x1 = ds[idx1]
    assert isinstance(x0, torch.Tensor) and isinstance(x1, torch.Tensor)
    assert x0.dtype == torch.float32 and x1.dtype == torch.float32
    assert x0.ndim == 3 and x0.shape[0] == 1  # grayscale channel-first
    # Values in [-1, 1]
    assert torch.all(x0 >= -1.0) and torch.all(x0 <= 1.0)
    assert torch.all(x1 >= -1.0) and torch.all(x1 <= 1.0)
    # Specific mapping checks
    assert torch.isclose(x0.min(), torch.tensor(-1.0))
    assert torch.isclose(x1.max(), torch.tensor(1.0), atol=1e-6)


def test_mnist_dataset_extension_filtering(tmp_path: Path):
    root = tmp_path
    (root / "2").mkdir(parents=True)
    # One PNG (counted), one JPG (ignored with default extension)
    _write_img(root / "2" / "c.png", 128)
    _write_img(root / "2" / "d.jpg", 128)

    ds_default = MNISTDataset(dataset_split="train", data_root=str(root))
    assert len(ds_default) == 1

    # With image_extension override, both can be loaded separately
    ds_jpg = MNISTDataset(dataset_split="train", data_root=str(root), image_extension="jpg")
    assert len(ds_jpg) == 1


def test_mnist_dataset_dataloader(tmp_path: Path):
    root = tmp_path
    (root / "3").mkdir(parents=True)
    for i in range(6):
        _write_img(root / "3" / f"{i}.png", i * 40)

    ds = MNISTDataset(dataset_split="train", data_root=str(root))
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)

    batch = next(iter(loader))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape[0] == 4
    assert batch.shape[1] == 1  # channels


def test_mnist_dataset_from_repo_data_if_available():
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "data" / "MNIST" / "mnist_images"
    if not default_root.exists():
        pytest.skip("Default MNIST data folder not present; skipping")

    ds = MNISTDataset(dataset_split="train", data_root=str(default_root))
    assert len(ds) > 0
    # Ensure at least some labels in 0..9
    assert any((y in range(10)) for y in ds.labels)
    x0 = ds[0]
    assert isinstance(x0, torch.Tensor)
    assert x0.shape[0] == 1
