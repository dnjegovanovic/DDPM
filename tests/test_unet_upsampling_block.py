import pytest
import torch

from ddpm_model.models.UNetBlocks import UpSamplingBlock
from ddpm_model.models.TimeEmbedding import time_embedding_fun


DEVICE_LIST = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICE_LIST)
def test_upsampling_block_no_skip_no_attn(device: str):
    B, C_in, C_out, H, W = 2, 16, 32, 16, 16
    x = torch.randn(B, C_in, H, W, device=device)

    block = UpSamplingBlock(
        in_channels=C_in,
        out_channels=C_out,
        skip_channels=0,
        time_emb_dim=None,
        num_heads=4,
        up_sample=True,
        custom_mha=True,
        num_layers=1,
        use_attn=False,
        grp_norm_chanels=8,
    ).to(device)

    y = block(x, time_emb=None, out_down=None)
    assert y.shape == (B, C_out, H * 2, W * 2)
    assert y.device.type == device


@pytest.mark.parametrize("device", DEVICE_LIST)
def test_upsampling_block_no_upsample(device: str):
    B, C_in, C_out, H, W = 2, 16, 32, 16, 16
    x = torch.randn(B, C_in, H, W, device=device)

    block = UpSamplingBlock(
        in_channels=C_in,
        out_channels=C_out,
        skip_channels=0,
        time_emb_dim=None,
        num_heads=4,
        up_sample=False,
        custom_mha=True,
        num_layers=1,
        use_attn=False,
        grp_norm_chanels=8,
    ).to(device)

    y = block(x)
    assert y.shape == (B, C_out, H, W)


@pytest.mark.parametrize("custom_mha", [True, False])
@pytest.mark.parametrize("heads", [1, 2, 4, 8])
@pytest.mark.parametrize("device", DEVICE_LIST)
def test_upsampling_block_with_attention_and_time_emb(device: str, heads: int, custom_mha: bool):
    B, C_in, C_out, H, W = 1, 16, 32, 16, 16
    time_dim = 128

    x = torch.randn(B, C_in, H, W, device=device)
    t = torch.arange(B, dtype=torch.float32, device=device)
    t_emb = time_embedding_fun(t, time_embedding_dim=time_dim)

    block = UpSamplingBlock(
        in_channels=C_in,
        out_channels=C_out,
        skip_channels=0,
        time_emb_dim=time_dim,
        num_heads=heads,
        up_sample=False,
        custom_mha=custom_mha,
        num_layers=2,
        use_attn=True,
        grp_norm_chanels=8,
    ).to(device)

    y = block(x, time_emb=t_emb)
    assert y.shape == (B, C_out, H, W)
    assert y.device.type == device


def test_upsampling_block_bad_groupnorm_raises():
    # out_channels must be divisible by grp_norm_chanels
    with pytest.raises(Exception):
        _ = UpSamplingBlock(
            in_channels=15,  # not divisible by 8 groups
            out_channels=30,  # not divisible by 8 groups
            skip_channels=0,
            time_emb_dim=None,
            num_heads=2,
            up_sample=True,
            custom_mha=True,
            num_layers=1,
            use_attn=True,
            grp_norm_chanels=8,
        )


@pytest.mark.parametrize("device", DEVICE_LIST)
def test_upsampling_block_skip_mismatch_raises(device: str):
    # The current implementation expects resnet input channels to match block.in_channels
    # Concatenating skip features increases channels and should fail without an adapter.
    B, C_in, C_out, H, W = 1, 16, 32, 8, 8
    x = torch.randn(B, C_in, H, W, device=device)
    skip = torch.randn(B, C_out, H * 2, W * 2, device=device)  # after upsample, sizes match

    block = UpSamplingBlock(
        in_channels=C_in,
        out_channels=C_out,
        skip_channels=0,
        time_emb_dim=None,
        num_heads=2,
        up_sample=True,
        custom_mha=True,
        num_layers=1,
        use_attn=False,
        grp_norm_chanels=8,
    ).to(device)

    with pytest.raises(Exception):
        _ = block(x, time_emb=None, out_down=skip)
