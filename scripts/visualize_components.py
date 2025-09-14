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
from ddpm_model.models.UNetBlocks import DownSamplingBlock


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


def main():
    parser = argparse.ArgumentParser(description="Visualize components: time embedding, attention, UNet downsample block")
    parser.add_argument("--outdir", type=Path, default=Path("artifacts"), help="Directory to save images")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("--skip", nargs="*", default=[], choices=["time", "mhsa", "mhca", "down"], help="Components to skip")

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

    print(f"Saved visualizations to {args.outdir}")


if __name__ == "__main__":
    main()
