"""AYS schedule on a UniPC predictor-corrector core.

This is a learned-schedule baseline for the report (Round 5b).  The
schedule is the Gaussian-closed-form, hierarchically subdivided AYS
schedule (see `_ays_grid.py`); the solver core is the same 2nd-order
UniPC predictor-corrector used by `unipc` and `proposed_unipc`, so the
only thing that changes between `unipc`, `proposed_unipc`, and
`ays_unipc` is the noise/grid schedule.

The 40-step AYS schedule is computed once (analytic, ~1 sec) and cached
to disk keyed on (sigma_min, sigma_max, c).  At sampling time we
piecewise log-linearly interpolate down to K+1 sigmas.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from ._ays_grid import (
    AysSchedule,
    ays_descending_sigmas,
    optimize_ays_40step,
)
from ._common import denoise, resolve_shape, resolve_sigma_range, sample_initial_noise
from .base import Sampler, SamplerOutput, register_sampler


def _cache_path(root: Path, sigma_min: float, sigma_max: float, c: float) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    tag = f"sm{sigma_min:.4f}_sM{sigma_max:.4f}_c{c:.3f}"
    return root / f"ays_40step_{tag}.json"


def _load_or_build_ays(root: Path, sigma_min: float, sigma_max: float, c: float,
                       force: bool = False) -> AysSchedule:
    path = _cache_path(root, sigma_min, sigma_max, c)
    if path.exists() and not force:
        d = json.loads(path.read_text())
        return AysSchedule(
            t_grid=np.array(d["t_grid"], dtype=np.float64),
            c2=float(d["c2"]),
            meta=d.get("meta", {}),
        )
    sched = optimize_ays_40step(sigma_min, sigma_max, c=c)
    path.write_text(json.dumps({
        "t_grid": sched.t_grid.tolist(),
        "c2": sched.c2,
        "meta": sched.meta,
    }, indent=2))
    return sched


@register_sampler("ays_unipc")
class AYSUniPC(Sampler):
    """UniPC 2nd-order predictor-corrector on an AYS schedule."""

    def __init__(self, *, cache_root: str | Path = "outputs/calibration",
                 c: float | None = None, force_rebuild: bool = False):
        self.cache_root = Path(cache_root)
        if c is None:
            c = float(os.environ.get("AD_AYS_C", "0.5"))
        self.c = float(c)
        self.force_rebuild = force_rebuild

    def _get_schedule(self, sigma_min: float, sigma_max: float) -> AysSchedule:
        return _load_or_build_ays(self.cache_root, sigma_min, sigma_max, self.c,
                                  force=self.force_rebuild)

    def sample(self, *, net, num_samples, num_steps, seed,
               device="cuda", batch_size=64, image_shape=None):
        device = torch.device(device)
        sigma_min, sigma_max = resolve_sigma_range(net)
        shape = resolve_shape(net, image_shape)
        ays = self._get_schedule(sigma_min, sigma_max)
        sigmas_np = ays_descending_sigmas(ays, num_steps)
        sigmas = torch.tensor(sigmas_np, dtype=torch.float32, device=device)

        out: list[torch.Tensor] = []
        nfe_per_sample = 0
        done = 0
        while done < num_samples:
            b = min(batch_size, num_samples - done)
            x = sample_initial_noise((b, *shape), float(sigmas[0]),
                                     seed=seed + done, device=device)
            cur_nfe = 0
            denoised_cur = denoise(net, x, sigmas[0])
            cur_nfe += 1
            denoised_prev = None

            for i in range(num_steps):
                sigma_i = sigmas[i]
                sigma_next = sigmas[i + 1]
                if sigma_next.item() == 0:
                    x = denoised_cur
                    break
                if denoised_prev is None:
                    h_i = sigma_i.log() - sigma_next.log()
                    x_pred = (sigma_next / sigma_i) * x - torch.expm1(-h_i) * denoised_cur
                else:
                    sigma_prev = sigmas[i - 1]
                    h_i_1 = sigma_prev.log() - sigma_i.log()
                    h_i = sigma_i.log() - sigma_next.log()
                    r = h_i_1 / h_i
                    D_pred = (1 + 1 / (2 * r)) * denoised_cur - (1 / (2 * r)) * denoised_prev
                    x_pred = (sigma_next / sigma_i) * x - torch.expm1(-h_i) * D_pred
                denoised_next = denoise(net, x_pred, sigma_next)
                cur_nfe += 1
                D_corr = 0.5 * denoised_cur + 0.5 * denoised_next
                h_i_eff = sigma_i.log() - sigma_next.log()
                x = (sigma_next / sigma_i) * x - torch.expm1(-h_i_eff) * D_corr
                denoised_prev = denoised_cur
                denoised_cur = denoised_next

            out.append(x.clamp(-1, 1).cpu())
            if done == 0:
                nfe_per_sample = cur_nfe
            done += b

        return SamplerOutput(
            samples=torch.cat(out, dim=0)[:num_samples],
            nfe=nfe_per_sample,
            metadata={
                "schedule": "ays_gaussian_closed_form",
                "solver": "unipc_2pc",
                "num_steps": num_steps,
                "ays_c": self.c,
                "ays_klub_40": ays.meta.get("klub_40"),
                "step_sigmas": sigmas_np.tolist(),
            },
        )
