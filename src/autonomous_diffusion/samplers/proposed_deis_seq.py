"""DEIS tAB-2 on the finite-N calibrated certificate grid (PDF 1 v4.1, Appendix C).

This is the native-DEIS sequence-coupled schedule: the grid minimizes the
MEASURED one-step DEIS error (Definition 1), optimized on-the-fly so the
surrogate has no exploitable blind spots (v4.1 Proposition 3.1). It
replaces the heuristic of borrowing Heun's pointwise d_s used by
`proposed_deis` (which exists because the leading-order Theorem-B
optimizer underperformed).

NFE per sample = num_steps (K substantive sigmas + 0 boundary).
"""
from __future__ import annotations

from pathlib import Path

import torch

from ._common import denoise, resolve_shape, sample_initial_noise
from ._seq_calib import get_or_build_deis_seq_grid
from .base import Sampler, SamplerOutput, register_sampler


@register_sampler("proposed_deis_seq")
class ProposedDEISSeq(Sampler):
    def __init__(self, *, cache_root: str | Path = "outputs/calibration"):
        self.cache_root = Path(cache_root)

    def sample(self, *, net, num_samples, num_steps, seed,
               device="cuda", batch_size=64, image_shape=None):
        device = torch.device(device)
        shape = resolve_shape(net, image_shape)
        grid_np = get_or_build_deis_seq_grid(
            net, num_steps, root=self.cache_root, device=device, image_shape=image_shape,
        )
        ss = torch.tensor(grid_np, dtype=torch.float32, device=device)
        n_int = ss.shape[0] - 1

        out = []
        nfe_per_sample = 0
        done = 0
        while done < num_samples:
            b = min(batch_size, num_samples - done)
            x = sample_initial_noise((b, *shape), float(ss[0]), seed=seed + done, device=device)
            prev = None
            cur = 0
            for i in range(n_int):
                si = ss[i]; sn = ss[i + 1]
                den = denoise(net, x, si); cur += 1
                eps_i = (x - den) / si
                if sn.item() == 0:
                    x = den
                    break
                if prev is None:
                    x = x + (sn - si) * eps_i
                else:
                    sp = ss[i - 1]
                    li = -si.log(); ln = -sn.log(); lp = -sp.log()
                    h_i = ln - li; h_p = li - lp
                    cc = 1 + h_i / (2 * h_p); cp = -h_i / (2 * h_p)
                    x = x + (sn - si) * (cc * eps_i + cp * prev)
                prev = eps_i
            out.append(x.clamp(-1, 1).cpu())
            if done == 0:
                nfe_per_sample = cur
            done += b

        return SamplerOutput(
            samples=torch.cat(out, dim=0)[:num_samples],
            nfe=nfe_per_sample,
            metadata={
                "solver": "deis_tAB2_on_calibrated_seq_grid",
                "schedule": "finite_N_calibrated_certificate_v4.1",
                "num_steps": num_steps,
                "step_sigmas": grid_np.tolist(),
            },
        )
