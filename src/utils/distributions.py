from __future__ import annotations

import math
import random
from typing import Optional, Dict, Any


def _clamp(value: float, min_value: Optional[float], max_value: Optional[float]) -> float:
    if min_value is not None:
        value = max(value, min_value)
    if max_value is not None:
        value = min(value, max_value)
    return value


def sample_bounded_int(params: Dict[str, Any], rng: Optional[random.Random] = None) -> int:
    """
    Sample a non-negative integer from a specified distribution and clamp to [min, max].

    Supported params:
      - type: "normal" | "poisson" | "exponential" | "uniform"
      - For normal: mean, stddev
      - For poisson: lambda
      - For exponential: rate (lambda) or scale (1/rate)
      - For uniform: low, high (inclusive bounds before clamping)
      - min, max: hard bounds after sampling
    """
    if rng is None:
        rng = random.Random()

    dist_type = str(params.get("type", "poisson")).lower()

    # bounds
    min_bound = params.get("min")
    max_bound = params.get("max")

    x: float
    if dist_type == "normal":
        mean = float(params.get("mean", 1.0))
        stddev = float(params.get("stddev", max(1e-9, params.get("stddev", 1.0))))
        x = rng.gauss(mean, stddev)
    elif dist_type == "poisson":
        lam = float(params.get("lambda", params.get("lam", 1.0)))
        # Knuth algorithm for small lam; for larger lam, normal approximation is acceptable
        if lam <= 30.0:
            L = math.exp(-lam)
            k = 0
            p = 1.0
            while p > L:
                k += 1
                p *= rng.random()
            x = k - 1
        else:
            # normal approximation N(lam, lam)
            x = rng.gauss(lam, math.sqrt(lam))
    elif dist_type == "exponential":
        if "rate" in params:
            rate = float(params["rate"]) or 1e-9
            x = rng.expovariate(rate)
        else:
            scale = float(params.get("scale", 1.0))
            rate = 1.0 / max(1e-9, scale)
            x = rng.expovariate(rate)
    elif dist_type == "uniform":
        low = float(params.get("low", 0.0))
        high = float(params.get("high", 1.0))
        if high < low:
            low, high = high, low
        x = rng.uniform(low, high)
    else:
        # default to poisson with lambda=1
        lam = float(params.get("lambda", 1.0))
        if lam <= 30.0:
            L = math.exp(-lam)
            k = 0
            p = 1.0
            while p > L:
                k += 1
                p *= rng.random()
            x = k - 1
        else:
            x = rng.gauss(lam, math.sqrt(lam))

    # clamp and round
    x = _clamp(x, min_bound, max_bound)
    # convert to int with floor, but ensure >= 0 after bounds
    value_int = int(math.floor(max(0.0, x)))
    # second clamp in case bounds are integers and floor overshoots
    if min_bound is not None:
        value_int = max(int(min_bound), value_int)
    if max_bound is not None:
        value_int = min(int(max_bound), value_int)
    return value_int


def sample_replica_count(role: str, dist_cfg: Dict[str, Any], rng: Optional[random.Random] = None) -> int:
    """
    Sample replicas per node for a role ("server" or "client") using a normal distribution
    configuration containing mean/stddev/min/max fields specific to the role.
    Fallback to 0 if config is incomplete.
    """
    if rng is None:
        rng = random.Random()

    if role not in ("server", "client"):
        return 0

    # map fields
    if role == "server":
        mean = dist_cfg.get("mean_per_server")
        std = dist_cfg.get("stddev_per_server")
        min_v = dist_cfg.get("min_per_server", 0)
        max_v = dist_cfg.get("max_per_server")
    else:
        mean = dist_cfg.get("mean_per_client")
        std = dist_cfg.get("stddev_per_client")
        min_v = dist_cfg.get("min_per_client", 0)
        max_v = dist_cfg.get("max_per_client")

    if mean is None or std is None:
        return 0

    return sample_bounded_int({
        "type": "normal",
        "mean": float(mean),
        "stddev": float(std) if float(std) > 0 else 1e-9,
        "min": int(min_v) if min_v is not None else 0,
        "max": int(max_v) if max_v is not None else None,
    }, rng)


