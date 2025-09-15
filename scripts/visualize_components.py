#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

# Ensure project root is on sys.path when running as a script (python scripts/visualize_components.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddpm_model.models.TimeEmbedding import time_embedding_fun
from ddpm_model.models.MultiHeadAttention import (
    MultiHeadSelfAttention,
    MultiHeadCrossAttention,
)
from ddpm_model.models.UNetBlocks import DownSamplingBlock, BottleNeck, UpSamplingBlock
from ddpm_model.models.UNet import UNet
from ddpm_model.datasets.MnistDatasets import MNISTDataset


def _ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def visualize_time_embedding(outdir: Path, device: str = "cpu"):
    t = torch.tensor([0.0, 1.0, 10.0, 100.0], device=device)
    dim = 320
    emb = time_embedding_fun(t, time_embedding_dim=dim).cpu()
    # Heatmap across dimensions for each timestep
    plt.figure(figsize=(10, 2.5))
    plt.imshow(emb, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(label="value")
    plt.yticks(range(len(t)), [f"t={int(x.item())}" for x in t])
    plt.xlabel("embedding dim")
    plt.title("Time Embedding Heatmap (sin|cos)")
    _savefig(outdir / "time_embedding_heatmap.png")

    # Plot first 64 dims for each timestep as line plot
    plt.figure(figsize=(10, 3))
    for i in range(len(t)):
        plt.plot(emb[i, :64].numpy(), label=f"t={int(t[i].item())}")
    plt.legend()
    plt.title("Time Embedding (first 64 dims)")
    plt.xlabel("dim")
    plt.ylabel("value")
    _savefig(outdir / "time_embedding_first64.png")


@torch.no_grad()
def _compute_mhsa_attention_weights(attn: MultiHeadSelfAttention, x: torch.Tensor, causal: bool):
    # x: (B,L,E)
    B, L, E = x.shape
    H = attn.num_heads
    head_dim = attn.head_dim

    q, k, v = attn.input_proj(x).chunk(3, dim=-1)
    q = q.view(B, L, H, head_dim).transpose(1, 2)
    k = k.view(B, L, H, head_dim).transpose(1, 2)

    weight = q @ k.transpose(-1, -2)  # (B,H,L,L)
    if causal:
        mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
        weight.masked_fill_(mask, -torch.inf)
    weight = weight / (head_dim ** 0.5)
    weight = torch.softmax(weight, dim=-1)
    return weight.cpu()


def visualize_mhsa(outdir: Path, device: str = "cpu", E: int = 8, H: int = 2, L: int = 8):
    x = torch.randn(1, L, E, device=device)
    attn = MultiHeadSelfAttention(num_heads=H, embed_dim=E).to(device).eval()

    w_nomask = _compute_mhsa_attention_weights(attn, x, causal=False)[0]  # (H,L,L)
    w_mask = _compute_mhsa_attention_weights(attn, x, causal=True)[0]

    for h in range(H):
        plt.figure(figsize=(7, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(w_nomask[h], vmin=0, vmax=1, cmap="magma")
        plt.title(f"Self-Attn H{h} (no mask)")
        plt.xlabel("key idx")
        plt.ylabel("query idx")
        plt.colorbar(fraction=0.046)

        plt.subplot(1, 2, 2)
        plt.imshow(w_mask[h], vmin=0, vmax=1, cmap="magma")
        plt.title(f"Self-Attn H{h} (causal)")
        plt.xlabel("key idx")
        plt.ylabel("query idx")
        plt.colorbar(fraction=0.046)

        _savefig(outdir / f"mhsa_head{h}.png")


@torch.no_grad()
def _compute_mhca_attention_weights(attn: MultiHeadCrossAttention, q: torch.Tensor, c: torch.Tensor):
    # q: (B,Lq,E), c: (B,Lkv,C)
    B, Lq, E = q.shape
    H = attn.num_heads
    head_dim = attn.head_dim

    q_proj = attn.query_proj(q)
    k_proj = attn.key_proj(c)

    qh = q_proj.view(B, Lq, H, head_dim).transpose(1, 2)  # (B,H,Lq,head)
    kh = k_proj.view(B, -1, H, head_dim).transpose(1, 2)  # (B,H,Lkv,head)

    weight = qh @ kh.transpose(-1, -2)  # (B,H,Lq,Lkv)
    weight = weight / (head_dim ** 0.5)
    weight = torch.softmax(weight, dim=-1)
    return weight.cpu()


def visualize_mhca(outdir: Path, device: str = "cpu", E: int = 8, H: int = 2, Lq: int = 4, Lkv: int = 6, cross_dim: int = 12):
    q = torch.randn(1, Lq, E, device=device)
    c = torch.randn(1, Lkv, cross_dim, device=device)
    attn = MultiHeadCrossAttention(num_heads=H, embed_dim=E, cross_dim=cross_dim).to(device).eval()

    w = _compute_mhca_attention_weights(attn, q, c)[0]  # (H,Lq,Lkv)
    for h in range(H):
        plt.figure(figsize=(5, 4))
        plt.imshow(w[h], vmin=0, vmax=1, cmap="magma")
        plt.title(f"Cross-Attn H{h}")
        plt.xlabel("key idx (context)")
        plt.ylabel("query idx")
        plt.colorbar(fraction=0.046)
        _savefig(outdir / f"mhca_head{h}.png")


def visualize_downsample_block(outdir: Path, device: str = "cpu"):
    B, C_in, C_out, H, W = 1, 16, 16, 32, 32
    # Simple gradient input for more interpretable feature maps
    base = torch.linspace(0, 1, steps=H * W, device=device).view(1, 1, H, W)
    x = base.repeat(B, C_in, 1, 1)

    time_dim = 128
    t = torch.tensor([10.0], device=device)
    t_emb = time_embedding_fun(t, time_embedding_dim=time_dim)

    block = DownSamplingBlock(
        in_channels=C_in,
        out_channels=C_out,
        time_emb_dim=time_dim,
        num_heads=2,
        down_sample=True,
        custom_mha=True,
        num_layers=2,
        use_attn=True,
        grp_norm_chanels=8,
    ).to(device).eval()

    with torch.no_grad():
        y = block(x, time_emb=t_emb)

    # Visualize a single channel before/after
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(x[0, 0].cpu(), cmap="gray")
    plt.title(f"Input C0 ({H}x{W})")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(y[0, 0].cpu(), cmap="gray")
    plt.title(f"Output C0 ({y.shape[-2]}x{y.shape[-1]})")
    plt.axis("off")

    _savefig(outdir / "downsample_block_channel0.png")


def visualize_bottleneck_block(outdir: Path, device: str = "cpu"):
    B, C_in, C_out, H, W = 1, 16, 16, 16, 16
    # Input pattern for interpretability
    xv, yv = torch.meshgrid(
        torch.linspace(-1, 1, steps=H, device=device),
        torch.linspace(-1, 1, steps=W, device=device),
        indexing="ij",
    )
    base = (xv + yv)[None, None]  # shape (1,1,H,W)
    x = base.repeat(B, C_in, 1, 1)

    time_dim = 128
    t = torch.tensor([25.0], device=device)
    t_emb = time_embedding_fun(t, time_embedding_dim=time_dim)

    block = BottleNeck(
        in_channels=C_in,
        out_channels=C_out,
        time_emb_dim=time_dim,
        num_heads=4,
        custom_mha=True,
        num_layers=2,
        grp_norm_chanels=8,
    ).to(device).eval()

    with torch.no_grad():
        y = block(x, time_emb=t_emb)

    # Visualize channel 0 before/after
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(x[0, 0].cpu(), cmap="gray")
    plt.title(f"Bottleneck Input C0 ({H}x{W})")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(y[0, 0].cpu(), cmap="gray")
    plt.title(f"Bottleneck Output C0 ({H}x{W})")
    plt.axis("off")

    _savefig(outdir / "bottleneck_block_channel0.png")


def visualize_upsample_block(outdir: Path, device: str = "cpu"):
    B, C_in, C_out, H, W = 1, 16, 16, 16, 16
    # Create a 2D sinusoidal pattern for visibility
    ys = torch.linspace(0, 2 * torch.pi, steps=H, device=device)
    xs = torch.linspace(0, 2 * torch.pi, steps=W, device=device)
    yv, xv = torch.meshgrid(ys, xs, indexing="ij")
    base = (torch.sin(xv) + torch.cos(yv))[None, None]
    x = base.repeat(B, C_in, 1, 1)

    time_dim = 128
    t = torch.tensor([5.0], device=device)
    t_emb = time_embedding_fun(t, time_embedding_dim=time_dim)

    block = UpSamplingBlock(
        in_channels=C_in,
        out_channels=C_out,
        skip_channels=0,
        time_emb_dim=time_dim,
        num_heads=2,
        up_sample=True,
        custom_mha=True,
        num_layers=2,
        use_attn=True,
        grp_norm_chanels=8,
    ).to(device).eval()

    with torch.no_grad():
        y = block(x, time_emb=t_emb)

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(x[0, 0].cpu(), cmap="viridis")
    plt.title(f"Upsample Input C0 ({H}x{W})")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(y[0, 0].cpu(), cmap="viridis")
    plt.title(f"Upsample Output C0 ({y.shape[-2]}x{y.shape[-1]})")
    plt.axis("off")

    _savefig(outdir / "upsample_block_channel0.png")


def visualize_unet(outdir: Path, device: str = "cpu"):
    # Minimal consistent UNet config matching GroupNorm groups of 8
    params = {
        "down_channels": [16, 32, 64],
        "mid_channels": [64, 64, 32],
        "down_sample": [True, True],
        "im_channels": 3,
        "time_emb_dim": 128,
        "num_down_layers": 1,
        "num_mid_layers": 1,
        "num_up_layers": 1,
    }

    model = UNet(UnetParams=params).to(device).eval()

    B, C, H, W = 1, params["im_channels"], 32, 32
    # Simple RGB gradient image
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, steps=H, device=device),
        torch.linspace(0, 1, steps=W, device=device),
        indexing="ij",
    )
    img = torch.stack([grid_x, grid_y, 0.5 * torch.ones_like(grid_x)], dim=0)[None]

    with torch.no_grad():
        out = model(img, t=10)

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(img[0].permute(1, 2, 0).cpu().clamp(0, 1))
    plt.title("UNet Input")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    out_vis = out[0].permute(1, 2, 0).cpu()
    out_vis = (out_vis - out_vis.min()) / (out_vis.max() - out_vis.min() + 1e-8)
    plt.imshow(out_vis)
    plt.title("UNet Output")
    plt.axis("off")

    _savefig(outdir / "unet_io.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize components: time embedding, attention, UNet, and dataset samples")
    parser.add_argument("--outdir", type=Path, default=Path("artifacts"), help="Directory to save images")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("--skip", nargs="*", default=[], choices=["time", "mhsa", "mhca", "down", "bottleneck", "upsample", "unet", "mnist"], help="Components to skip")
    parser.add_argument("--mnist-root", type=Path, default=None, help="Path to MNIST-like folder structure (digit subfolders)")

    args = parser.parse_args()
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    _ensure_outdir(args.outdir)

    if "time" not in args.skip:
        visualize_time_embedding(args.outdir, device=device)
    if "mhsa" not in args.skip:
        visualize_mhsa(args.outdir, device=device)
    if "mhca" not in args.skip:
        visualize_mhca(args.outdir, device=device)
    if "down" not in args.skip:
        visualize_downsample_block(args.outdir, device=device)
    # visualize bottleneck unless skipped
    if "bottleneck" not in args.skip:
        visualize_bottleneck_block(args.outdir, device=device)
    if "upsample" not in args.skip:
        visualize_upsample_block(args.outdir, device=device)
    if "unet" not in args.skip:
        visualize_unet(args.outdir, device=device)
    # Default to repo data/MNIST/mnist_images if not provided and it exists
    default_mnist = PROJECT_ROOT / "data" / "MNIST" / "mnist_images"
    mnist_root = args.mnist_root if args.mnist_root is not None else (default_mnist if default_mnist.exists() else None)

    if "mnist" not in args.skip and mnist_root is not None and mnist_root.exists():
        ds = MNISTDataset(dataset_split="viz", data_root=str(mnist_root))
        # Grid of first N samples
        n = min(16, len(ds))
        if n > 0:
            cols = 4
            rows = (n + cols - 1) // cols
            plt.figure(figsize=(cols * 2, rows * 2))
            for i in range(n):
                x = ds[i].cpu()
                vis = (x + 1) / 2.0  # back to [0,1]
                plt.subplot(rows, cols, i + 1)
                plt.imshow(vis[0], cmap="gray")
                plt.axis("off")
            _savefig(args.outdir / "mnist_grid.png")

        # Label histogram if labels present
        if hasattr(ds, "labels") and len(ds.labels) > 0:
            plt.figure(figsize=(6, 3))
            counts = [0] * 10
            for y in ds.labels:
                if isinstance(y, int) and 0 <= y <= 9:
                    counts[y] += 1
            plt.bar(list(range(10)), counts)
            plt.xlabel("digit")
            plt.ylabel("count")
            plt.title("MNIST label histogram")
            _savefig(args.outdir / "mnist_label_hist.png")

    print(f"Saved visualizations to {args.outdir}")


if __name__ == "__main__":
    main()
