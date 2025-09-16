import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage

# Optional live plotting; training continues if unavailable
try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False

from ddpm_model.datasets.MnistDatasets import MNISTDataset
from ddpm_model.models.UNet import UNet
from ddpm_model.models.LinearNoiseScheduler import LinearNoiseScheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_x0_from_pred(scheduler: LinearNoiseScheduler, x_t: torch.Tensor, noise_pred: torch.Tensor, t_scalar: int) -> torch.Tensor:
    """Compute x0 prediction using scheduler buffers (mirrors sample_prev_timestep logic)."""
    t = int(t_scalar)
    return (
        (1.0 / scheduler.sqrt_alphas_cumprod[t])
        * (x_t - noise_pred * scheduler.sqrt_one_minus_alphas_cumprod[t])
    ).clamp(-1.0, 1.0)


@torch.no_grad()
def _visualize_denoising_progress(
    model: UNet,
    scheduler: LinearNoiseScheduler,
    clean_images: torch.Tensor,
    t_scalar: int,
    outdir: Path,
    step_tag: str,
    live: bool = False,
    fixed_noise: torch.Tensor | None = None,
):
    """Save (and optionally live-show) a grid: clean | noisy_t | denoised(x0_pred).

    - Expects images in [-1, 1].
    - Uses a single fixed timestep `t_scalar` for clarity and speed.
    """
    model.eval()
    B = clean_images.shape[0]

    # Prepare noise and create noisy images at timestep t
    device = clean_images.device
    noise = fixed_noise if fixed_noise is not None else torch.randn_like(clean_images)
    t_vec = torch.full((B,), int(t_scalar), device=device, dtype=torch.long)
    x_t = scheduler.add_noise(clean_images, noise, t_vec)

    # Predict noise and compute x0
    noise_pred = model(x_t, int(t_scalar))
    x0_pred = _compute_x0_from_pred(scheduler, x_t, noise_pred, int(t_scalar))

    # Convert to [0,1] for visualization
    def to_01(x: torch.Tensor) -> torch.Tensor:
        return ((x.clamp(-1.0, 1.0) + 1.0) / 2.0).detach().cpu()

    clean_01 = to_01(clean_images)
    noisy_01 = to_01(x_t)
    den_01 = to_01(x0_pred)

    # Stack rows: [clean ...], [noisy ...], [denoised ...]
    grid_tensor = torch.cat([clean_01, noisy_01, den_01], dim=0)
    # nrow = batch size so rows become [clean, noisy, denoised]
    grid = make_grid(grid_tensor, nrow=B)

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"denoise_progress_{step_tag}.png"
    save_image(grid, out_path)

    # Optionally live show
    if live and _HAS_PLT:
        plt.ion()
        plt.figure("DDPM Denoising Progress", figsize=(6, 6))
        img = ToPILImage()(grid)
        plt.imshow(img)
        plt.title(f"Clean | Noisy@t={t_scalar} | Denoised — {step_tag}")
        plt.axis("off")
        plt.pause(0.001)
        plt.clf()
    model.train()


try:
    # Import from scripts if available (during repo runs)
    from scripts.make_gif import make_training_gif as _make_training_gif
except Exception:
    _make_training_gif = None


def train(config, args):
    mnist_trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=None
    )

    scheduler = LinearNoiseScheduler(device, **vars(config.ddpm_model))

    # Create the dataset
    mnist = MNISTDataset("train", data_root=args.data_path)
    mnist_loader = DataLoader(
        mnist, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    # Instantiate the model
    model = UNet(**vars(config.unet_model)).to(device)
    model.train()

    output_dir = Path(args.task_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if found
    if args.ckpt_file:
        ckpt_file_path = Path(args.ckpt_file)
        ckpt_files = list(ckpt_file_path.glob("*.ckpt"))

        if ckpt_files:
            print("Loading existing model...")
            model.load_state_dict(
                torch.load(ckpt_file_path / ckpt_files[0], map_location=device)
            )

    # Specify training parameters
    num_epochs = args.num_epochs
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    # Visualization setup
    viz_live = bool(getattr(args, "viz_live", False))
    viz_save = bool(getattr(args, "viz_save", False))
    viz_enabled = viz_live or viz_save
    viz_interval = int(getattr(args, "viz_interval", 500))
    viz_num = int(getattr(args, "viz_num", 8))
    viz_t_raw = int(getattr(args, "viz_t", -1))
    viz_t = (
        scheduler.num_timesteps - 1
        if (viz_t_raw < 0 or viz_t_raw >= scheduler.num_timesteps)
        else viz_t_raw
    )
    viz_outdir = output_dir / "viz"

    # Fix a small batch for consistent visualization
    fixed_loader = DataLoader(mnist, batch_size=viz_num, shuffle=False, num_workers=0)
    fixed_batch = next(iter(fixed_loader)).float().to(device) if viz_enabled else None
    fixed_noise = torch.randn_like(fixed_batch) if viz_enabled else None

    # Run training
    global_step = 0
    for epoch_idx in range(num_epochs):
        losses = []
        for im in tqdm(mnist_loader):
            optimizer.zero_grad()
            im = im.float().to(device)

            # Sample random noise
            noise = torch.randn_like(im).to(device)

            # Sample timestep
            t = torch.randint(0, scheduler.num_timesteps, (im.shape[0],)).to(device)

            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            # Periodic visualization of denoising progress on a fixed batch
            if viz_enabled and (global_step % viz_interval == 0):
                _visualize_denoising_progress(
                    model,
                    scheduler,
                    fixed_batch,
                    viz_t,
                    viz_outdir,
                    step_tag=f"ep{epoch_idx+1}_it{global_step}",
                    live=viz_live,
                    fixed_noise=fixed_noise,
                )

            global_step += 1

        print(
            "Finished epoch:{} | Loss : {:.4f}".format(
                epoch_idx + 1,
                np.mean(losses),
            )
        )
        torch.save(
            model.state_dict(), os.path.join(args.task_name, args.ckpt_save_name)
        )

        # End-of-epoch snapshot
        if viz_enabled:
            _visualize_denoising_progress(
                model,
                scheduler,
                fixed_batch,
                viz_t,
                viz_outdir,
                step_tag=f"ep{epoch_idx+1}_end",
                live=viz_live,
                fixed_noise=fixed_noise,
            )

    print("Done Training ...")
    # Assemble a training progress GIF from saved PNGs if any were saved
    if viz_enabled and _make_training_gif is not None:
        _make_training_gif(viz_outdir)


def eval_help(model: UNet, scheduler: LinearNoiseScheduler, args, cfg):
    """Run ancestral sampling and save progressive x0 predictions.

    - Uses current UNet and scheduler to denoise from x_T to x_0.
    - Saves x0_pred grids at each step into `task_name/samples/`.
    """
    unet_params = cfg.unet_model.UnetParams
    C = int(unet_params["im_channels"])  # channels
    H = int(unet_params["im_size"])      # height
    W = int(unet_params["im_size"])      # width

    xt = torch.randn((args.num_samples, C, H, W), device=device)

    samples_dir = Path(args.task_name) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(reversed(range(scheduler.num_timesteps))):
        # Predict noise and step back
        noise_pred = model(xt, int(i))
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, int(i))

        # Save predicted x0 at this step (mapped to [0,1])
        vis = (x0_pred.clamp(-1.0, 1.0) + 1.0) / 2.0
        grid = make_grid(vis.detach().cpu(), nrow=args.num_grid_rows)
        save_image(grid, samples_dir / f"x0_{i:04d}.png")


def eval(args, cfg):
    """Load a trained model and generate samples, saving progressive x0 grids."""
    # Resolve checkpoint path
    ckpt_path = None
    if getattr(args, "ckpt_file", None):
        p = Path(args.ckpt_file)
        if p.is_file():
            ckpt_path = p
        elif p.is_dir():
            # pick first .pth or .ckpt in directory
            cand = sorted(list(p.glob("*.pth")) + list(p.glob("*.ckpt")))
            if cand:
                ckpt_path = cand[0]
    if ckpt_path is None:
        ckpt_path = Path(args.task_name) / getattr(args, "ckpt_save_name", "test_001.pth")

    # Load model
    model = UNet(**vars(cfg.unet_model)).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Create the noise scheduler and run sampling
    scheduler = LinearNoiseScheduler(device, **vars(cfg.ddpm_model))
    with torch.no_grad():
        eval_help(model, scheduler, args, cfg)
