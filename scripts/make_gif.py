#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from PIL import Image


def make_training_gif(
    viz_dir: Path,
    pattern: str = "denoise_progress_*.png",
    out_name: str = "progress.gif",
    duration_ms: int = 200,
) -> Optional[Path]:
    """Create an animated GIF from saved training visualization PNGs.

    - Collects images matching `pattern` in `viz_dir`, sorted by name.
    - Saves `out_name` in the same directory.
    - Returns the output path if created, otherwise None.
    """
    viz_dir = Path(viz_dir)
    if not viz_dir.exists():
        return None

    frames = sorted(viz_dir.glob(pattern))
    if not frames:
        return None

    images = [Image.open(p).convert("RGB") for p in frames]
    base = images[0]
    append = images[1:] if len(images) > 1 else images
    out_path = viz_dir / out_name
    try:
        base.save(
            out_path,
            save_all=True,
            append_images=append,
            duration=duration_ms,
            loop=0,
            format="GIF",
        )
        return out_path
    finally:
        for im in images:
            try:
                im.close()
            except Exception:
                pass


def _parse_args():
    p = argparse.ArgumentParser(description="Assemble training progress GIF from PNGs")
    p.add_argument("viz_dir", type=Path, help="Directory with denoise_progress_*.png files")
    p.add_argument("--pattern", type=str, default="denoise_progress_*.png", help="Glob pattern for frames")
    p.add_argument("--out-name", type=str, default="progress.gif", help="Output GIF filename")
    p.add_argument("--duration-ms", type=int, default=200, help="Frame duration in milliseconds")
    return p.parse_args()


def main():
    args = _parse_args()
    out = make_training_gif(args.viz_dir, args.pattern, args.out_name, args.duration_ms)
    if out is None:
        print("No frames found; GIF not created.")
    else:
        print(f"Saved GIF to: {out}")


if __name__ == "__main__":
    main()

