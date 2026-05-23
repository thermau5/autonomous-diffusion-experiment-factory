"""Clean-FID wrapper.

We use GaParmar/clean-fid so resize/quantization choices do not drift FID
across baselines. The wrapper supports two modes:

  - folder mode: writes samples to a temp PNG dir and calls compute_fid(folder, ref)
  - stat-cache mode: uses cleanfid's built-in `dataset_name` reference stats
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch


@dataclass
class CleanFIDConfig:
    dataset_name: str          # "cifar10", "ffhq", "afhqv2"
    dataset_res: int           # 32 or 64
    dataset_split: str = "train"
    mode: str = "clean"        # "clean" | "legacy_pytorch" | "legacy_tensorflow"
    num_workers: int = 4
    batch_size: int = 64


def _to_uint8_bchw(samples: torch.Tensor | np.ndarray) -> np.ndarray:
    x = torch.as_tensor(samples)
    if x.dtype == torch.uint8:
        arr = x.cpu().numpy()
    else:
        x = x.detach().float().cpu()
        # accept [-1,1] or [0,1]
        if float(x.min()) < -1e-4:
            x = (x.clamp(-1, 1) + 1) * 0.5
        arr = (x.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).numpy()
    if arr.ndim != 4 or arr.shape[1] not in (1, 3):
        raise ValueError(f"expected [N,C,H,W] with C in {{1,3}}, got {arr.shape}")
    return arr


def _dump_pngs(arr_bchw: np.ndarray, out_dir: Path) -> None:
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(arr_bchw.shape[0]):
        img = arr_bchw[i].transpose(1, 2, 0)
        if img.shape[2] == 1:
            img = img[:, :, 0]
            Image.fromarray(img, mode="L").save(out_dir / f"{i:07d}.png")
        else:
            Image.fromarray(img, mode="RGB").save(out_dir / f"{i:07d}.png")


def compute_clean_fid(
    samples: torch.Tensor | np.ndarray,
    cfg: CleanFIDConfig,
    *,
    tmp_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compute Clean-FID of `samples` against the named reference dataset.

    Returns {fid, n_samples, mode, dataset, resolution}. Uses cleanfid's
    bundled reference statistics when available.
    """
    try:
        from cleanfid import fid as clean_fid_mod
    except ImportError as e:
        raise RuntimeError(
            "clean-fid is not installed. `pip install clean-fid` (already in environment.yml)."
        ) from e

    arr = _to_uint8_bchw(samples)
    n = arr.shape[0]

    tmp_root = Path(tmp_root) if tmp_root else None
    with tempfile.TemporaryDirectory(prefix="cleanfid-", dir=tmp_root) as td:
        out_dir = Path(td) / "samples"
        _dump_pngs(arr, out_dir)
        fid_val = clean_fid_mod.compute_fid(
            str(out_dir),
            dataset_name=cfg.dataset_name,
            dataset_res=cfg.dataset_res,
            dataset_split=cfg.dataset_split,
            mode=cfg.mode,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
        )
    return {
        "fid": float(fid_val),
        "n_samples": int(n),
        "mode": cfg.mode,
        "dataset": cfg.dataset_name,
        "resolution": cfg.dataset_res,
        "split": cfg.dataset_split,
    }
