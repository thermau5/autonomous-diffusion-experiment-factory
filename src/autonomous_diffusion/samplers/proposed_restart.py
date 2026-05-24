"""Proposed control on a Restart wrapper around an EDM-Heun ODE core.

Restart is a *wrapper* protocol -- it takes a base ODE solver, integrates
to some sigma_lo, injects noise to come back to sigma_hi, re-integrates,
and loops K times before a tail to zero. The certificate's m* is applied
to the BASE solver's grid (sigma_max -> sigma_lo). The restart interval
[sigma_lo, sigma_hi] and inner sub-grid are kept at their reference Xu et
al CIFAR-10 defaults.

NFE = base (Heun) + K*(inner cycle Heun) + tail (Heun)
    = (2*num_steps - 1) + K*(2*inner_steps - 1) + (2*tail_steps - 1)
"""
from __future__ import annotations

from pathlib import Path

import torch

from ._common import (
    denoise,
    karras_sigmas,
    resolve_shape,
    resolve_sigma_range,
    sample_initial_noise,
)
from .base import Sampler, SamplerOutput, register_sampler
from .proposed_control import (
    calibrate,
    calibration_cache_path,
    load_calibration,
    optimal_step_sigmas,
    save_calibration,
)


def _heun_through(net, x, sigmas):
    nfe = 0
    n = sigmas.shape[0] - 1
    for i in range(n):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        denoised = denoise(net, x, sigma)
        d = (x - denoised) / sigma
        x_next = x + (sigma_next - sigma) * d
        nfe += 1
        if sigma_next.item() > 0:
            denoised2 = denoise(net, x_next, sigma_next)
            d2 = (x_next - denoised2) / sigma_next
            x_next = x + (sigma_next - sigma) * 0.5 * (d + d2)
            nfe += 1
        x = x_next
    return x, nfe


def _optimal_subgrid_in_range(calib, num_steps, sigma_lo, sigma_hi, *, p, k):
    """Reuse the certificate's CDF inversion but restricted to the
    [sigma_lo, sigma_hi] support. Picks `num_steps+1` sigmas inside that
    band (descending, sigma_hi -> sigma_lo)."""
    import numpy as np
    sigmas = calib.sigma_grid
    d = calib.d_per_interval
    intervals = np.abs(sigmas[:-1] - sigmas[1:])
    interval_mid = 0.5 * (sigmas[:-1] + sigmas[1:])
    interval_mid = np.clip(interval_mid, 1e-12, None)
    w = interval_mid ** (-k)
    weights = (np.clip(d, 1e-12, None) * w) ** (1.0 / (p + 1)) * intervals
    cum = np.concatenate([[0.0], np.cumsum(weights)])
    # Restrict to support [sigma_lo, sigma_hi] using monotonic interp.
    # sigmas is descending; cum is increasing with index.
    cum_at_hi = float(np.interp(sigma_hi, sigmas[::-1], cum[::-1]))
    cum_at_lo = float(np.interp(sigma_lo, sigmas[::-1], cum[::-1]))
    target = np.linspace(cum_at_hi, cum_at_lo, num_steps + 1)
    new = np.interp(target, cum, sigmas)
    new[0] = sigma_hi
    new[-1] = sigma_lo
    return new.astype("float32")


@register_sampler("proposed_restart")
class ProposedRestart(Sampler):
    """Restart wrapper with certificate-optimal grids for both the base
    trajectory and each restart cycle's inner subgrid.

    `num_steps` controls the BASE trajectory grid (sigma_max -> sigma_lo).
    The inner cycle and tail use Xu et al CIFAR-10 defaults: K=1 restart,
    inner_steps=6, tail_steps=4, sigma_band=[0.06, 1.0].
    """
    def __init__(
        self,
        *,
        cache_root: str | Path = "outputs/calibration",
        num_calib_samples: int = 16,
        num_intervals: int = 32,
        num_ref_substeps: int = 16,
        calib_seed: int = 0xCA11B,
        force_recalibrate: bool = False,
        p: int = 2,
        perceptual_weight_k: float | None = None,
        num_restart: int = 1,
        inner_steps: int = 6,
        tail_steps: int = 4,
        sigma_lo: float = 0.06,
        sigma_hi: float = 1.0,
    ):
        import os
        self.cache_root = Path(cache_root)
        self.num_calib_samples = num_calib_samples
        self.num_intervals = num_intervals
        self.num_ref_substeps = num_ref_substeps
        self.calib_seed = calib_seed
        self.force_recalibrate = force_recalibrate
        self.p = p
        if perceptual_weight_k is None:
            perceptual_weight_k = float(os.environ.get("AD_PROPOSED_K", "2.0"))
        self.perceptual_weight_k = float(perceptual_weight_k)
        self.num_restart = num_restart
        self.inner_steps = inner_steps
        self.tail_steps = tail_steps
        self.sigma_lo = sigma_lo
        self.sigma_hi = sigma_hi

    def _get_calibration(self, net, device, image_shape):
        path = calibration_cache_path(net, root=self.cache_root)
        if path.exists() and not self.force_recalibrate:
            return load_calibration(path), 0, False
        calib = calibrate(
            net,
            num_calib_samples=self.num_calib_samples,
            num_intervals=self.num_intervals,
            num_ref_substeps=self.num_ref_substeps,
            seed=self.calib_seed,
            device=device,
            image_shape=image_shape,
        )
        save_calibration(calib, path)
        traj_nfe = 2 * self.num_intervals - 1
        per_interval_nfe = 2 + (2 * self.num_ref_substeps - 1)
        total_calib_nfe = self.num_calib_samples * (traj_nfe + self.num_intervals * per_interval_nfe)
        return calib, total_calib_nfe, True

    def sample(self, *, net, num_samples, num_steps, seed, device="cuda", batch_size=64, image_shape=None):
        device = torch.device(device)
        sigma_min, sigma_max = resolve_sigma_range(net)
        shape = resolve_shape(net, image_shape)
        sigma_lo = max(self.sigma_lo, sigma_min)
        sigma_hi = min(self.sigma_hi, sigma_max)
        assert sigma_lo < sigma_hi

        calib, calib_nfe, did_recalibrate = self._get_calibration(net, device, image_shape)

        # Certificate-optimal sub-grid for the BASE trajectory (sigma_max -> sigma_lo).
        base_sub = _optimal_subgrid_in_range(
            calib, num_steps, sigma_lo=sigma_lo, sigma_hi=sigma_max,
            p=self.p, k=self.perceptual_weight_k,
        )
        base_sigmas = torch.tensor(base_sub, dtype=torch.float32, device=device)

        # Inner Karras band for each restart cycle (kept at Xu et al defaults).
        inner_full = karras_sigmas(self.inner_steps, sigma_lo, sigma_hi, device=device).to(torch.float32)
        inner_sigmas = inner_full[:-1]   # stop at sigma_lo, no trailing zero
        # Tail sigma_lo -> 0
        tail_sigmas = karras_sigmas(self.tail_steps, sigma_min, sigma_lo, device=device).to(torch.float32)

        var_inj = sigma_hi ** 2 - sigma_lo ** 2
        std_inj = float(max(0.0, var_inj)) ** 0.5

        out = []
        nfe_per_sample = 0
        done = 0
        while done < num_samples:
            b = min(batch_size, num_samples - done)
            x = sample_initial_noise((b, *shape), float(base_sigmas[0]),
                                     seed=seed + done, device=device)
            gen_restart = torch.Generator(device=device).manual_seed(int(seed) + int(done) + 31337)
            cur_nfe = 0
            x, used = _heun_through(net, x, base_sigmas)
            cur_nfe += used
            for _k in range(self.num_restart):
                if std_inj > 0:
                    noise = torch.randn(x.shape, generator=gen_restart, device=device, dtype=x.dtype)
                    x = x + std_inj * noise
                x, used = _heun_through(net, x, inner_sigmas)
                cur_nfe += used
            x, used = _heun_through(net, x, tail_sigmas)
            cur_nfe += used
            out.append(x.clamp(-1, 1).cpu())
            if done == 0:
                nfe_per_sample = cur_nfe
            done += b

        amortised = calib_nfe / max(num_samples, 1)
        return SamplerOutput(
            samples=torch.cat(out, dim=0)[:num_samples],
            nfe=nfe_per_sample,
            metadata={
                "solver": "restart_heun_on_optimal_base_sigmas",
                "num_steps_base": num_steps,
                "p": self.p,
                "perceptual_weight_k": self.perceptual_weight_k,
                "base_step_sigmas": base_sub.tolist(),
                "sigma_band": [sigma_lo, sigma_hi],
                "num_restart": self.num_restart,
                "inner_steps": self.inner_steps,
                "tail_steps": self.tail_steps,
                "calibration_nfe_total": calib_nfe,
                "calibration_nfe_amortized": amortised,
                "did_recalibrate": did_recalibrate,
                "calib_meta": calib.meta,
            },
        )
