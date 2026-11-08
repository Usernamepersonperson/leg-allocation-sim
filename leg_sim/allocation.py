"""Per-interval allocation of community PV production to consumers.

All quantities are integer Wh per 15-minute interval, shapes ``loads (T, n)`` and ``production (T,)``.
Invariants for every method (checked in tests):

* ``0 <= alloc <= loads`` (a member never receives more than it consumes)
* ``alloc.sum(axis=1) <= production`` and ``production - alloc.sum(axis=1)`` is the export, so
  ``alloc.sum + export == production`` holds exactly in every interval.
"""

from __future__ import annotations

import numpy as np

METHODS = ("static", "dynamic", "fair")


def _check(loads: np.ndarray, production: np.ndarray) -> None:
    if loads.ndim != 2 or production.ndim != 1 or loads.shape[0] != production.shape[0]:
        raise ValueError(f"shape mismatch: loads {loads.shape}, production {production.shape}")
    if (loads < 0).any() or (production < 0).any():
        raise ValueError("negative energy values are not allowed")


def _round_exact(a: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """Round float allocations to whole Wh per interval (largest remainder), preserving the interval total."""
    a = np.clip(a, 0, cap)
    fl = np.floor(a)
    frac = a - fl
    r = np.rint(frac.sum(axis=1)).astype(np.int64)
    order = np.argsort(-frac, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(a.shape[0])[:, None]
    rank[rows, order] = np.arange(a.shape[1])[None, :]
    bump = (rank < r[:, None]) & (frac > 1e-9)
    return (fl + bump).astype(np.int64)


def allocate_static(loads: np.ndarray, production: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """Fixed allocation key: member i may use ``shares[i] * production``; its unused part is exported."""
    _check(loads, production)
    shares = np.asarray(shares, dtype=float)
    if shares.shape != (loads.shape[1],) or (shares < 0).any() or not np.isclose(shares.sum(), 1.0):
        raise ValueError("static shares must be non-negative, one per member, and sum to 1")
    ent = production[:, None].astype(float) * shares[None, :]
    return _round_exact(np.minimum(loads.astype(float), ent), loads)


def allocate_dynamic(loads: np.ndarray, production: np.ndarray) -> np.ndarray:
    """Dynamic key proportional to each member's consumption in the interval."""
    _check(loads, production)
    tot = loads.sum(axis=1).astype(float)
    ratio = np.divide(production, tot, out=np.zeros_like(tot), where=tot > 0)
    return _round_exact(loads * np.minimum(1.0, ratio)[:, None], loads)


def allocate_fair(loads: np.ndarray, production: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted max-min fair allocation (progressive filling).

    Each member is entitled to ``weights[i]/sum(weights) * production``. Entitlement a member cannot use is
    re-offered to the members that still want more, again in proportion to the weights, until either all
    demand is met or the production is used up. No member ever gets less than ``min(load, entitlement)`` and
    no production is wasted while someone still consumes.
    """
    _check(loads, production)
    w = np.asarray(weights, dtype=float)
    if w.shape != (loads.shape[1],) or (w <= 0).any():
        raise ValueError("fair weights must be positive, one per member")
    c = loads.astype(float)
    alloc = np.zeros_like(c)
    rem = production.astype(float).copy()
    active = c > 0
    for _ in range(c.shape[1] + 1):
        w_act = (w[None, :] * active).sum(axis=1)
        live = (w_act > 0) & (rem > 0)
        if not live.any():
            break
        lam = np.divide(rem, w_act, out=np.zeros_like(rem), where=w_act > 0)
        give = w[None, :] * lam[:, None]
        need = c - alloc
        sat = active & (need <= give + 1e-9) & live[:, None]
        any_sat = sat.any(axis=1)
        # rows where some members saturate: give them exactly their need, continue with the rest
        alloc = np.where(sat, alloc + need, alloc)
        rem = rem - np.where(sat, need, 0.0).sum(axis=1)
        active = active & ~sat
        # rows where nobody saturates: everyone left takes its proportional share and the row is done
        done = live & ~any_sat
        alloc = np.where(done[:, None] & active, alloc + give, alloc)
        rem = np.where(done, 0.0, rem)
        active = active & ~done[:, None]
    return _round_exact(alloc, loads)


def allocate(method: str, loads: np.ndarray, production: np.ndarray, shares: np.ndarray) -> np.ndarray:
    if method == "static":
        return allocate_static(loads, production, shares)
    if method == "dynamic":
        return allocate_dynamic(loads, production)
    if method == "fair":
        return allocate_fair(loads, production, shares)
    raise ValueError(f"unknown allocation method {method!r}; choose from {METHODS}")


def static_shares(annual_kwh: np.ndarray) -> np.ndarray:
    """Static key proportional to expected yearly consumption."""
    a = np.asarray(annual_kwh, dtype=float)
    if (a < 0).any() or a.sum() <= 0:
        raise ValueError("annual consumption must be non-negative with a positive total")
    return a / a.sum()
