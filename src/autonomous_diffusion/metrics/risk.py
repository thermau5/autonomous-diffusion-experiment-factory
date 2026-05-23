"""Risk-violation metric V(u) (placeholder for the Pareto-3 axis).

The certificate uses a per-sample risk R_hat_s(u) and a deviation bound
B_n(u, delta) such that the cumulative violation rate is

    V(u) = (1/S) sum_{s=1}^{S}  1{ R_hat_s(u) > epsilon }.

A real per-sample R_hat requires either a per-sample log-density estimate
(NLL variant) or a per-sample Wasserstein contribution (W variant), both
of which are expensive on real images. For the first sweep we report a
proxy: the per-seed FID minus the matched-budget baseline FID, expressed
as a violation when it exceeds an FID-budget epsilon. This is NOT the
theoretical certificate; it's an honest placeholder so the metric is wired
end-to-end and the locked-test pipeline doesn't error on a missing field.

When the certificate's actual R_hat is implemented later (Round 3?), this
file is the one place that needs to change.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def fid_proxy_violation_rate(
    per_seed_fids: Iterable[float],
    *,
    baseline_fid: float,
    epsilon: float,
) -> dict:
    """Fraction of seeds whose FID exceeds (baseline_fid + epsilon).

    Returns a dict carrying the rate, threshold, and per-seed booleans for
    auditability. This is a *proxy* for the certificate's R_hat and is
    labeled as such in the output JSON.
    """
    fids = np.array(list(per_seed_fids), dtype=np.float64)
    threshold = float(baseline_fid + epsilon)
    violations = fids > threshold
    return {
        "kind": "fid_proxy",
        "threshold": threshold,
        "baseline_fid": float(baseline_fid),
        "epsilon": float(epsilon),
        "num_seeds": int(fids.size),
        "num_violations": int(violations.sum()),
        "violation_rate": float(violations.mean()) if fids.size else 0.0,
        "per_seed_violation": violations.tolist(),
        "per_seed_fid": fids.tolist(),
        "caveat": "Proxy only; certificate-true R_hat not yet implemented.",
    }
