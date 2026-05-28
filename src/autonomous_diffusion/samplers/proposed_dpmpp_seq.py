"""DPM-Solver++ 2M on the finite-N calibrated certificate grid (PDF v4.1, App. C).

Companion to the locked `proposed_dpmpp` (pointwise m_s*). This runs the
same DPM-Solver++ 2M solver on the calibrated *sequence* grid (the grid
that minimizes the measured one-step 2M-predictor error). Reporting both
quantifies the A approximately B gap for this strictly-multistep core.

NFE per sample = num_steps.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ._common import resolve_sigma_range, run_sampler
from ._seq_calib import get_or_build_seq_grid
from .base import Sampler, SamplerOutput, register_sampler
from .dpm_solver_pp import _dpmpp_update


@register_sampler("proposed_dpmpp_seq")
class ProposedDPMPPSeq(Sampler):
    CORE = "dpmpp"

    def __init__(self, *, cache_root: str | Path = "outputs/calibration"):
        self.cache_root = Path(cache_root)

    def sample(self, *, net, num_samples, num_steps, seed,
               device="cuda", batch_size=64, image_shape=None):
        grid = get_or_build_seq_grid(net, num_steps, self.CORE, root=self.cache_root,
                                     device=device, image_shape=image_shape)
        sigmas = torch.tensor(grid, dtype=torch.float32, device=device)
        samples, nfe = run_sampler(
            net=net, sigmas=sigmas, update_fn=_dpmpp_update,
            num_samples=num_samples, seed=seed, device=device,
            batch_size=batch_size, image_shape=image_shape,
            state_factory=lambda: {"prev_denoised": None},
        )
        return SamplerOutput(samples=samples, nfe=nfe,
                             metadata={"solver": "dpm_solver_pp_2M_on_calibrated_seq_grid",
                                       "schedule": "finite_N_calibrated_certificate_v4.1",
                                       "num_steps": num_steps, "order": 2,
                                       "step_sigmas": grid.tolist()})
