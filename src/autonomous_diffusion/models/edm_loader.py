"""Load pretrained EDM (Karras 2022) networks.

EDM ships the generator as a pickled `EDMPrecond` whose unpickling pulls
`torch_utils` and `dnnlib` from the EDM source tree. We vendor that tree under
`third_party/edm/` and prepend it to sys.path before unpickling.

Pretrained URLs are the official NVlabs CDN paths.
"""
from __future__ import annotations

import os
import pickle
import sys
import urllib.request
from pathlib import Path
from typing import Any

import torch


EDM_CHECKPOINTS: dict[str, str] = {
    "cifar10":  "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl",
    "ffhq":     "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl",
    "afhqv2":   "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-vp.pkl",
}


def _repo_root() -> Path:
    # src/autonomous_diffusion/models/edm_loader.py  ->  repo root
    return Path(__file__).resolve().parents[3]


def _ensure_edm_on_path() -> Path:
    edm_dir = _repo_root() / "third_party" / "edm"
    if not (edm_dir / "torch_utils").is_dir() or not (edm_dir / "dnnlib").is_dir():
        raise RuntimeError(
            f"third_party/edm is missing torch_utils/ or dnnlib/. "
            f"Run `make vendor_edm` to clone NVlabs/edm into {edm_dir}."
        )
    p = str(edm_dir)
    if p not in sys.path:
        sys.path.insert(0, p)
    return edm_dir


def _cache_dir() -> Path:
    d = _repo_root() / "third_party" / "edm_ckpts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length", "0"))
        read = 0
        chunk = 1 << 20
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            fh.write(buf)
            read += len(buf)
            if total:
                pct = 100 * read / total
                print(f"  ... {pct:5.1f}%  ({read/1e6:.1f} / {total/1e6:.1f} MB)", end="\r", flush=True)
        print()
    os.replace(tmp, dest)


def load_edm_network(
    dataset: str,
    device: str | torch.device = "cuda",
    *,
    eval_mode: bool = True,
) -> Any:
    """Return the pretrained EDM EDMPrecond network on `device`.

    The returned object exposes `.sigma_min`, `.sigma_max`, `.sigma_data`,
    `.round_sigma(sigma)`, and is callable as `net(x, sigma, class_labels=None)`
    returning the denoised x0 estimate.
    """
    if dataset not in EDM_CHECKPOINTS:
        raise KeyError(f"unknown dataset {dataset!r}; choose from {list(EDM_CHECKPOINTS)}")
    _ensure_edm_on_path()

    url = EDM_CHECKPOINTS[dataset]
    ckpt_path = _cache_dir() / Path(url).name
    _download(url, ckpt_path)

    with open(ckpt_path, "rb") as fh:
        data = pickle.load(fh)
    net = data["ema"] if isinstance(data, dict) and "ema" in data else data
    net = net.to(device)
    if eval_mode:
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
    return net
