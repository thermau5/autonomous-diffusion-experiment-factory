"""DPM-Solver-v3 (Zheng, Lu et al., NeurIPS 2023) on the EDM CIFAR-10 backbone.

Wraps the official `third_party/dpm_solver_v3/codebases/edm/samplers/dpm_solver_v3.py`
predictor-corrector loop, using the paper-team-released EMS coefficients
\((l, s, b)\) for `edm-cifar10-32x32-uncond-vp` from
\texttt{statistics/edm-cifar10-32x32-uncond-vp/}.

This is a Level-1 baseline (new solver core), not a Level-2 baseline:
the schedule is fully separable from the EMS-corrected update law.

The companion `proposed_dpm_solver_v3` (separate file) applies our
certificate-optimal grid \(m^\star_s\) on top of the same EMS-corrected
update law.

NFE accounting: DPM-Solver-v3 is a multistep solver, 1 NFE per step,
so total NFE per sample = num_steps.

Defaults follow the paper's EDM CIFAR-10 sample.sh:
  skip_type = "logSNR"
  order     = 3
  p_pseudo  = (steps <= 5)
  use_corrector = (steps <= 6)
  c_pseudo  = False
  lower_order_final = True
  stats_dir = "0.002_80.0_1200_1024" for steps < 10, "0.002_80.0_120_4096" else
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from ._common import resolve_shape, resolve_sigma_range
from .base import Sampler, SamplerOutput, register_sampler


REPO_ROOT = Path(__file__).resolve().parents[3]
DPMV3_DIR = REPO_ROOT / "third_party" / "dpm_solver_v3" / "codebases" / "edm"
DPMV3_STATS_ROOT = REPO_ROOT / "third_party" / "dpm_solver_v3" / "statistics" / "statistics" / "edm-cifar10-32x32-uncond-vp"


def _ensure_path():
    p = str(DPMV3_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _select_stats_dir(num_steps: int) -> Path:
    """Per paper's sample.sh: high-resolution stats for low NFE, low-res for high NFE."""
    if num_steps < 10:
        return DPMV3_STATS_ROOT / "0.002_80.0_1200_1024"
    return DPMV3_STATS_ROOT / "0.002_80.0_120_4096"


@register_sampler("dpm_solver_v3")
class DPMSolverV3(Sampler):
    """DPM-Solver-v3 with paper-released EMS on the EDM CIFAR-10 backbone."""

    def __init__(self, *, order: int = 3, skip_type: str = "logSNR",
                 degenerated: bool = False):
        self.order = int(order)
        self.skip_type = str(skip_type)
        self.degenerated = bool(degenerated)

    def sample(self, *, net, num_samples, num_steps, seed,
               device="cuda", batch_size=64, image_shape=None):
        _ensure_path()
        from samplers.dpm_solver_v3 import DPM_Solver_v3
        from samplers.utils import NoiseScheduleEDM

        device = torch.device(device)
        shape = resolve_shape(net, image_shape)
        sigma_min, sigma_max = resolve_sigma_range(net)

        stats_dir = _select_stats_dir(num_steps)
        if not stats_dir.is_dir():
            raise FileNotFoundError(
                f"DPM-Solver-v3 EMS stats not found at {stats_dir}; "
                "ensure third_party/dpm_solver_v3/statistics is downloaded."
            )

        ns = NoiseScheduleEDM()
        dpm_v3 = DPM_Solver_v3(
            str(stats_dir),
            ns,
            steps=num_steps,
            t_start=float(sigma_max),
            t_end=float(sigma_min),
            skip_type=self.skip_type,
            device=str(device),
            degenerated=self.degenerated,
        )

        # Their pipeline expects model(x, t_input, cond) -> x_0 prediction.
        # Our EDM net takes (x, sigma_batch) -> denoised x_0 directly.
        def edm_x0_model(x, t_input, cond=None):
            if t_input.dim() == 0:
                t_b = t_input.expand(x.shape[0]).to(x.dtype)
            else:
                t_b = t_input.to(x.dtype)
            return net(x, t_b).to(x.dtype)

        from samplers.utils import model_wrapper
        model_fn = model_wrapper(edm_x0_model, ns, class_labels=None)

        out: list[torch.Tensor] = []
        nfe_per_sample = 0
        done = 0
        while done < num_samples:
            b = min(batch_size, num_samples - done)
            g = torch.Generator(device=device).manual_seed(int(seed + done))
            # Paper's EDM example uses latents in float64 then * sigma_max; we
            # follow that for numerical fidelity to their reported FID.
            x_init = torch.randn(
                (b, *shape), generator=g, device=device, dtype=torch.float64
            ) * float(sigma_max)
            with torch.no_grad():
                x = dpm_v3.sample(
                    model_fn,
                    x_init,
                    order=self.order,
                    p_pseudo=(num_steps <= 5),
                    use_corrector=(num_steps <= 6),
                    c_pseudo=False,
                    lower_order_final=True,
                )
            out.append(x.clamp(-1, 1).float().cpu())
            if done == 0:
                nfe_per_sample = num_steps
            done += b

        return SamplerOutput(
            samples=torch.cat(out, dim=0)[:num_samples],
            nfe=nfe_per_sample,
            metadata={
                "solver": "dpm_solver_v3",
                "schedule": f"dpm_solver_v3_{self.skip_type}",
                "ems_stats_dir": stats_dir.name,
                "order": self.order,
                "p_pseudo": num_steps <= 5,
                "use_corrector": num_steps <= 6,
                "c_pseudo": False,
                "lower_order_final": True,
                "num_steps": num_steps,
            },
        )
