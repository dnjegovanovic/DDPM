from pathlib import Path
from setuptools import setup, find_packages


PACKAGE_NAME = "ddpm-model"
MODULE_NAME = "ddpm_model"
ROOT = Path(__file__).parent


def read_readme() -> str:
    readme_path = ROOT / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "DDPM with linear noise scheduler and UNet implementation."


def read_requirements(path: Path) -> list[str]:
    reqs = []
    if not path.exists():
        return reqs
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("-r ") or s.startswith("--requirement "):
            # support includes like "-r other.txt"
            inc = s.split(maxsplit=1)[1]
            reqs.extend(read_requirements((path.parent / inc).resolve()))
            continue
        reqs.append(s)
    return reqs


base_reqs = read_requirements(ROOT / "requirements" / "requirements.txt")
torch_reqs = read_requirements(ROOT / "requirements" / "torch.txt")
viz_reqs = read_requirements(ROOT / "requirements" / "viz.txt")
dev_reqs = read_requirements(ROOT / "requirements" / "dev.txt")


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    description="Denoising Diffusion Probabilistic Model (linear beta) with UNet",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Your Name",
    url="",
    license="MIT",
    packages=find_packages(exclude=("tests", "scripts", "artifacts")),
    include_package_data=True,
    package_data={
        MODULE_NAME: [
            "config.yml",
        ]
    },
    python_requires=">=3.9",
    install_requires=base_reqs,
    extras_require={
        # Install torch/torchvision explicitly depending on your platform (cpu/cuda)
        "torch": torch_reqs or ["torch", "torchvision"],
        "viz": viz_reqs or ["matplotlib"],
        "dev": dev_reqs or ["pytest", "black", "ruff"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # If you want a CLI entrypoint later, create ddpm_model/cli.py with a main() and uncomment:
    # entry_points={
    #     "console_scripts": [
    #         "ddpm-train=ddpm_model.cli:main",
    #     ],
    # },
)
