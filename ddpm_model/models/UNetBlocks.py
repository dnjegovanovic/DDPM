import torch
import torch.nn as nn

from .MultiHeadAttention import MultiHeadSelfAttention, MultiHeadCrossAttention
from .TimeEmbedding import TimeEmbedding, time_embedding_fun

"""
UNet Architecture with Attention Blocks for Diffusion Models

This module implements a UNet-like architecture with residual blocks, attention mechanisms,
and time embedding for diffusion models. It consists of downsampling blocks, a bottleneck,
and upsampling blocks with skip connections.
"""


class DownSamplingBlock(nn.Module):
    """
    A UNet-style downsampling block that combines:
    - Residual blocks with time embeddings
    - Optional attention mechanisms
    - Optional spatial downsampling
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int = None,
        num_heads: int = 4,
        down_sample: bool = True,
        custom_mha: bool = True,
        num_layers: int = 1,
        use_attn: bool = True,
        grp_norm_chanels: int = 8,
    ):
        super().__init__()

        # Store configuration parameters
        self.down_sample = down_sample  # Whether to reduce spatial dimensions
        self.custom_mha = custom_mha  # Use custom or PyTorch's MHA implementation
        self.num_layers = num_layers  # Number of residual/attention layers in block
        self.time_emb_dim = time_emb_dim
        self.use_attn = use_attn

        # First part of residual block (per layer)
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(
                        grp_norm_chanels, in_channels if i == 0 else out_channels
                    ),  # Normalize input
                    nn.SiLU(),  # Activation function
                    nn.Conv2d(  # Channel transformation
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(num_layers)
            ]
        )

        # Time embedding processing (per layer)
        if self.time_emb_dim:
            self.time_emb_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.SiLU(),  # Activation for time embedding
                        nn.Linear(
                            time_emb_dim, out_channels
                        ),  # Project time emb to channel space
                    )
                    for _ in range(num_layers)
                ]
            )

        # Second part of residual block (per layer)
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(grp_norm_chanels, out_channels),  # Normalization
                    nn.SiLU(),  # Activation
                    nn.Conv2d(  # Final convolution in residual path
                        out_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                )
                for _ in range(num_layers)
            ]
        )

        # Attention mechanism components (per layer)
        if self.use_attn:
            self.attention_norm = nn.ModuleList(
                [
                    nn.GroupNorm(
                        grp_norm_chanels, out_channels
                    )  # Normalization before attention
                    for _ in range(num_layers)
                ]
            )

            # Choose attention implementation
            if custom_mha:
                self.attention = nn.ModuleList(
                    [
                        MultiHeadSelfAttention(
                            num_heads, out_channels, input_proj_bias=False
                        )
                        for _ in range(num_layers)
                    ]
                )
            else:
                self.attention = nn.ModuleList(
                    [
                        nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                        for _ in range(num_layers)
                    ]
                )

        # Residual connection projection (per layer)
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(  # 1x1 conv for channel matching
                    in_channels if i == 0 else out_channels, out_channels, kernel_size=1
                )
                for i in range(num_layers)
            ]
        )

        # Final downsampling layer
        self.down_sample_conv = (
            nn.Conv2d(  # Spatial downsampling (halves resolution)
                out_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
            if self.down_sample
            else nn.Identity()  # Skip downsampling
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor = None) -> torch.Tensor:
        out = x  # Preserve original input for residual connection

        for i in range(self.num_layers):
            # Residual block processing
            resnet_input = out  # Store input for residual connection

            # First convolution path
            out = self.resnet_conv_first[i](out)

            if self.time_emb_dim:
                # Add time embedding (broadcasted to spatial dimensions)
                out = out + self.time_emb_layers[i](time_emb)[:, :, None, None]

            # Second convolution path
            out = self.resnet_conv_second[i](out)

            # Residual connection with projection
            out = out + self.residual_input_conv[i](resnet_input)

            # Attention processing
            if self.use_attn:
                batch_size, chanels, h, w = out.shape
                in_attn = out.reshape(
                    batch_size, chanels, h * w
                )  # Flatten spatial dims
                in_attn = self.attention_norm[i](in_attn)
                if self.custom_mha:
                    # Custom attention expects (batch, seq_len, channels)
                    in_attn = in_attn.transpose(-1, -2)
                    out_attn = self.attention[i](in_attn)
                    out_attn = out_attn.transpose(-1, -2)
                else:
                    # PyTorch MHA expects (batch, seq_len, channels)
                    in_attn = in_attn.transpose(1, 2)
                    out_attn, _ = self.attention[i](in_attn, in_attn, in_attn)
                    out_attn = out_attn.transpose(1, 2)

                # Reshape back to original dimensions
                out_attn = out_attn.reshape(batch_size, chanels, h, w)
                out = out + out_attn  # Add attention output

        # Final downsampling
        out = self.down_sample_conv(out)
        assert (
            out.ndim == 4
        ), "Donwsample output dont haher 4 dim [batch, channels, H, W]"
        return out