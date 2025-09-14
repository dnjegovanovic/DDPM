import pytest
import torch

from ddpm_model.models.UNetBlocks import DownSamplingBlock
from ddpm_model.models.TimeEmbedding import time_embedding_fun


DEVICE_LIST = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICE_LIST)
def test_downsampling_block_no_downsample_no_attn(device: str):
    B, C_in, C_out, H, W = 2, 16, 32, 16, 16
    x = torch.randn(B, C_in, H, W, device=device)

    block = DownSamplingBlock(
        in_channels=C_in,
        out_channels=C_out,
        time_emb_dim=None,
        num_heads=4,
        down_sample=False,
        custom_mha=True,
        num_layers=1,
        use_attn=False,
        grp_norm_chanels=8,
    ).to(device)

    y = block(x)
    assert y.shape == (B, C_out, H, W)
    assert y.device.type == device


@pytest.mark.parametrize("device", DEVICE_LIST)
def test_downsampling_block_with_downsample(device: str):
    B, C_in, C_out, H, W = 2, 16, 32, 32, 32
    x = torch.randn(B, C_in, H, W, device=device)

    block = DownSamplingBlock(
        in_channels=C_in,
        out_channels=C_out,
        time_emb_dim=None,
        num_heads=4,
        down_sample=True,
        custom_mha=True,
        num_layers=1,
        use_attn=False,
        grp_norm_chanels=8,
    ).to(device)

    y = block(x)
    assert y.shape == (B, C_out, H // 2, W // 2)


@pytest.mark.parametrize("custom_mha", [True, False])
@pytest.mark.parametrize("heads", [1, 2, 4, 8])
@pytest.mark.parametrize("device", DEVICE_LIST)
def test_downsampling_block_with_attention_and_time_emb(device: str, heads: int, custom_mha: bool):
    B, C_in, C_out, H, W = 1, 16, 32, 16, 16
    time_dim = 128

    # GroupNorm requires C_out % groups == 0; groups=8 by default -> 32 % 8 == 0
    x = torch.randn(B, C_in, H, W, device=device)
    # Use a simple increasing timestep vector to build a valid time embedding
    t = torch.arange(B, dtype=torch.float32, device=device)
    t_emb = time_embedding_fun(t, time_embedding_dim=time_dim)

    block = DownSamplingBlock(
        in_channels=C_in,
        out_channels=C_out,
        time_emb_dim=time_dim,
        num_heads=heads,
        down_sample=False,
        custom_mha=custom_mha,
        num_layers=2,
        use_attn=True,
        grp_norm_chanels=8,
    ).to(device)

    y = block(x, time_emb=t_emb)
    assert y.shape == (B, C_out, H, W)
    assert y.device.type == device


def test_downsampling_block_bad_groupnorm_raises():
    # out_channels must be divisible by grp_norm_chanels
    with pytest.raises(Exception):
        _ = DownSamplingBlock(
            in_channels=15,
            out_channels=30,  # not divisible by 8 groups
            time_emb_dim=None,
            num_heads=2,
            down_sample=False,
            custom_mha=True,
            num_layers=1,
            use_attn=True,
            grp_norm_chanels=8,
        )


@pytest.mark.parametrize("device", DEVICE_LIST)
def test_downsampling_block_missing_time_emb_raises(device: str):
    B, C_in, C_out, H, W = 1, 8, 16, 8, 8
    x = torch.randn(B, C_in, H, W, device=device)
    block = DownSamplingBlock(
        in_channels=C_in,
        out_channels=C_out,
        time_emb_dim=64,  # expects a time embedding
        num_heads=2,
        down_sample=False,
        custom_mha=True,
        num_layers=1,
        use_attn=False,
        grp_norm_chanels=8,
    ).to(device)

    with pytest.raises(Exception):
        _ = block(x, time_emb=None)

