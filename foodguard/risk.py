from __future__ import annotations

from typing import Dict


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def compute_crs(ps: float, ns: float, es: float, posture: str = "recall_high") -> Dict[str, float | str]:
    """Compute a Combined Risk Score (CRS) from PS/NS/ES.

    This is a simple, transparent weighted fusion baseline that can be
    calibrated later. We prioritize recall by default (higher weight on PS).

    Weights are placeholders; tune and calibrate off held-out data.
    """
    weights = {
        "recall_high": (0.6, 0.1, 0.3),
        "balanced": (0.5, 0.2, 0.3),
        "precision_high": (0.45, 0.25, 0.30),
    }
    w_ps, w_ns, w_es = weights.get(posture, weights["recall_high"])
    crs = clamp01(w_ps * ps + w_ns * ns + w_es * es)
    return {"crs": crs, "posture": posture}

