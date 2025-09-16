import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm

from ddpm_model.modules import train_ddpm as ddpm_modules
from ddpm_model.config.core import config

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Arguments for ddpm training", add_help=True
    )
    parser.add_argument(
        "--train-ddpm-model",
        action="store_true",
        help="Train ddpm model-false by default",
    )

    parser.add_argument(
        "--eval-ddpm-model",
        action="store_true",
        help="Inferenc ddpm model-false by default",
    )

    parser.add_argument("--task-name", help="name of task", default="default", type=str)
    parser.add_argument(
        "--data-path",
        help="path to data dir",
        type=str,
    )
    parser.add_argument("--ckpt-file", help="Existing model load", type=str)
    parser.add_argument(
        "--ckpt-save-name", help="Save model name", default="test_001.pth", type=str
    )

    parser.add_argument("--batch-size", help="Batch size", default="64", type=int)
    parser.add_argument("--num-epochs", help="number of epoch", default="40", type=int)
    parser.add_argument("--lr", help="Learning rate", default="1e-4", type=float)
    # ----------------------------------------------------------------------
    # Visualization during training
    viz_parser = parser.add_argument_group("Training Visualization")
    viz_parser.add_argument(
        "--viz-save",
        action="store_true",
        help="Save denoising progress PNGs to task_name/viz",
    )
    viz_parser.add_argument(
        "--viz-live",
        action="store_true",
        help="Show live matplotlib window with denoising progress",
    )
    viz_parser.add_argument(
        "--viz-interval",
        type=int,
        default=500,
        help="Steps between visualization snapshots",
    )
    viz_parser.add_argument(
        "--viz-num",
        type=int,
        default=8,
        help="Number of samples in the visualization grid",
    )
    viz_parser.add_argument(
        "--viz-t",
        type=int,
        default=-1,
        help="Timestep to visualize denoising from (-1 = last step)",
    )
    # ----------------------------------------------------------------------
    eval_ddpm_parser = parser.add_argument_group("Eval DDPM model")
    eval_ddpm_parser.add_argument(
        "--num-samples", help="number of samples", default="100", type=int
    )
    eval_ddpm_parser.add_argument(
        "--num-grid-rows",
        help="number of grids to visualize images",
        default="10",
        type=int,
    )
    # -----------------------------------------------------------------------
    args = parser.parse_args()

    if args.train_ddpm_model:
        ddpm_modules.train(config, args)
    elif args.eval_ddpm_model:
        ddpm_modules.eval(args, config)
