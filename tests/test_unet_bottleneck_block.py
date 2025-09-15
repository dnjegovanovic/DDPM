import pytest
import torch

from ddpm_model.models.UNetBlocks import BottleNeck
from ddpm_model.models.TimeEmbedding import time_embedding_fun


DEVICE_LIST = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("custom_mha", [True, False])
@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("device", DEVICE_LIST)
def test_bottleneck_shapes_and_device(device: str, heads: int, custom_mha: bool):
    B, C_in, C_out, H, W = 2, 16, 32, 16, 16
    x = torch.randn(B, C_in, H, W, device=device)
    time_dim = 128
    t = torch.arange(B, dtype=torch.float32, device=device)
    t_emb = time_embedding_fun(t, time_embedding_dim=time_dim)

    block = BottleNeck(
        in_channels=C_in,
        out_channels=C_out,
        time_emb_dim=time_dim,
        num_heads=heads,
        custom_mha=custom_mha,
        num_layers=2,
        grp_norm_chanels=8,
    ).to(device)

    y = block(x, time_emb=t_emb)

    assert y.shape == (B, C_out, H, W)
    assert y.device.type == device


@pytest.mark.parametrize("device", DEVICE_LIST)
def test_bottleneck_without_time_embedding(device: str):
    B, C_in, C_out, H, W = 1, 8, 16, 8, 8
    x = torch.randn(B, C_in, H, W, device=device)

    block = BottleNeck(
        in_channels=C_in,
        out_channels=C_out,
        time_emb_dim=None,
        num_heads=2,
        custom_mha=True,
        num_layers=1,
        grp_norm_chanels=8,
    ).to(device)

    y = block(x, time_emb=None)
    assert y.shape == (B, C_out, H, W)


@pytest.mark.parametrize("device", DEVICE_LIST)
def test_bottleneck_missing_time_embedding_raises(device: str):
    B, C_in, C_out, H, W = 1, 8, 16, 8, 8
    x = torch.randn(B, C_in, H, W, device=device)

    block = BottleNeck(
        in_channels=C_in,
        out_channels=C_out,
        time_emb_dim=64,  # expects time embedding provided
        num_heads=2,
        custom_mha=True,
        num_layers=1,
        grp_norm_chanels=8,
    ).to(device)

    with pytest.raises(Exception):
        _ = block(x, time_emb=None)

