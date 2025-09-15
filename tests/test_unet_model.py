import pytest
import torch

from ddpm_model.models.UNet import UNet


DEVICE_LIST = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def build_unet_params():
    return {
        "down_channels": [16, 32, 64],
        "mid_channels": [64, 64, 32],
        "down_sample": [True, True],
        "im_channels": 3,
        "time_emb_dim": 128,
        "num_down_layers": 1,
        "num_mid_layers": 1,
        "num_up_layers": 1,
    }


@pytest.mark.parametrize("device", DEVICE_LIST)
def test_unet_forward_shapes(device: str):
    params = build_unet_params()

    model = UNet(UnetParams=params).to(device)
    B, C, H, W = 2, params["im_channels"], 32, 32
    x = torch.randn(B, C, H, W, device=device)

    y = model(x, t=10)

    assert y.shape == (B, C, H, W)
    assert y.device.type == device


def test_unet_param_assertions():
    bad = build_unet_params()
    bad["mid_channels"] = [32, 64, 64]  # violates mid_channels[0] == down_channels[-1]
    with pytest.raises(AssertionError):
        _ = UNet(UnetParams=bad)

