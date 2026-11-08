"""Synthetic 15-minute load profiles (SIMULATED, not measured meter data).

Each generator builds a plausible shape (daily rhythm, weekday/weekend, season, temperature, random appliance
events), then scales it to a target yearly energy. Output is integer Wh per 15-minute interval, as a smart
meter would report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from leg_sim.opendata import _round_preserving, temperature_quarter_hourly

SLOTS_PER_DAY = 96


def _grid(year: int = 2023) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", periods=365 * SLOTS_PER_DAY, freq="15min")


def _g(h: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((h - mu) / sigma) ** 2)


def _season(doy: np.ndarray, amp: float) -> np.ndarray:
    return 1.0 + amp * np.cos(2 * np.pi * (doy - 20) / 365)


def _scale_to_wh(shape: np.ndarray, annual_kwh: float) -> np.ndarray:
    shape = np.clip(shape, 0, None)
    total = shape.sum()
    if total <= 0:
        raise ValueError("profile shape is all zero")
    return _round_preserving(shape * (annual_kwh * 1000.0 / total))


def _noise(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    """Log-normal multiplicative noise with mild autocorrelation (neighbouring quarters are similar)."""
    z = rng.normal(0, 1, n + 2)
    z = (z[:-2] + 2 * z[1:-1] + z[2:]) / np.sqrt(6)
    return np.exp(sigma * z - 0.5 * sigma**2)


def _events(rng, n_days: int, per_week: float, kw: float, dur_slots: tuple[int, int], hours: tuple[float, float]):
    """Random appliance cycles (washing, dishwasher...): returns Wh/quarter additions."""
    out = np.zeros(n_days * SLOTS_PER_DAY)
    for d in range(n_days):
        k = rng.poisson(per_week / 7)
        for _ in range(k):
            start = int(rng.uniform(*hours) * 4)
            dur = int(rng.integers(dur_slots[0], dur_slots[1] + 1))
            s = d * SLOTS_PER_DAY + start
            out[s : s + dur] += kw * 250.0  # kW * 0.25 h * 1000 -> Wh
    return out


def household(rng: np.random.Generator, annual_kwh: float, home_during_day: float, ev_kwh: float = 0.0) -> np.ndarray:
    """Household profile; ``home_during_day`` in [0, 1] scales the midday activity (0.3 = both away at work)."""
    idx = _grid()
    h = (idx.hour + idx.minute / 60).to_numpy(dtype=float)
    dow = idx.day_of_week.to_numpy()
    doy = idx.day_of_year.to_numpy()
    wk = dow < 5
    shift = rng.normal(0, 0.5)
    morning = np.where(wk, _g(h, 6.9 + shift, 0.8), _g(h, 8.8 + shift, 1.3))
    midday = _g(h, 12.3, 1.2) * np.where(wk, home_during_day, 0.6 + 0.4 * home_during_day)
    afternoon = _g(h, 15.5, 2.0) * np.where(wk, 0.4 * home_during_day, 0.5)
    evening = _g(h, 19.2 + shift, 1.7) * np.where(wk, 1.0, 1.15)
    shape = 0.22 + 0.9 * morning + 0.8 * midday + 0.3 * afternoon + 1.5 * evening
    shape *= _season(doy, 0.18) * _noise(rng, len(idx), 0.45)
    wh = shape * (annual_kwh * 1000.0 / shape.sum()) * 0.88
    days = 365
    wh += _events(rng, days, per_week=3.5, kw=1.6, dur_slots=(4, 8), hours=(7.5, 21.0))
    wh += _events(rng, days, per_week=2.0, kw=1.1, dur_slots=(4, 6), hours=(13.0, 21.5))
    prof = wh / wh.sum() * annual_kwh * 1000.0 if ev_kwh == 0 else wh
    out = prof
    if ev_kwh > 0:
        out = prof / prof.sum() * (annual_kwh - ev_kwh) * 1000.0 + ev_charging(rng, ev_kwh)
    return _round_preserving(out)


def ev_charging(rng: np.random.Generator, annual_kwh: float, power_kw: float = 7.4) -> np.ndarray:
    """Wh/quarter of a home wallbox: ~4 sessions a week, mostly evening arrival, some daytime (home office)."""
    out = np.zeros(365 * SLOTS_PER_DAY)
    sessions = int(annual_kwh / 11.0)  # average ~11 kWh per session
    days = rng.choice(365, size=min(sessions, 365), replace=False) if sessions <= 365 else rng.integers(0, 365, sessions)
    for d in days:
        e_kwh = float(np.clip(rng.normal(11, 4), 3, 30))
        if rng.random() < 0.15:
            start = int(rng.uniform(9, 13) * 4)  # home-office day
        else:
            start = int(rng.uniform(17.5, 21) * 4)
        left = e_kwh * 1000.0
        pos = d * SLOTS_PER_DAY + start
        while left > 0 and pos < len(out):
            step = min(power_kw * 250.0, left)
            out[pos] += step
            left -= step
            pos += 1
    return out * (annual_kwh * 1000.0 / max(out.sum(), 1.0))


def common_area(rng: np.random.Generator, annual_kwh: float) -> np.ndarray:
    """Stairwell lighting, lift, ventilation: near-flat with an evening/morning lighting bump."""
    idx = _grid()
    h = (idx.hour + idx.minute / 60).to_numpy(dtype=float)
    doy = idx.day_of_year.to_numpy()
    shape = (0.7 + 0.5 * _g(h, 7, 1.5) + 0.8 * _g(h, 19, 2.5)) * _season(doy, 0.12) * _noise(rng, len(idx), 0.12)
    return _scale_to_wh(shape, annual_kwh)


def central_heat_pump(rng: np.random.Generator, annual_kwh: float) -> np.ndarray:
    """Heat-pump electricity driven by PVGIS 2 m temperature (heating below ~15 C, small hot-water base)."""
    t = temperature_quarter_hourly().to_numpy()
    idx = _grid()
    h = (idx.hour + idx.minute / 60).to_numpy(dtype=float)
    daily = pd.Series(t).rolling(96, center=True, min_periods=1).mean().to_numpy()
    heating = np.clip(15.0 - daily, 0, None)
    diurnal = 0.75 + 0.35 * _g(h, 6.5, 1.8) + 0.3 * _g(h, 19, 2.2)
    shape = (heating * diurnal + 0.35 * (0.6 + 0.6 * _g(h, 6.5, 1.5) + 0.4 * _g(h, 19, 2.0))) * _noise(rng, len(idx), 0.08)
    return _scale_to_wh(shape, annual_kwh)


def office(rng: np.random.Generator, annual_kwh: float) -> np.ndarray:
    idx = _grid()
    h = (idx.hour + idx.minute / 60).to_numpy(dtype=float)
    wk = (idx.day_of_week.to_numpy() < 5).astype(float)
    t = temperature_quarter_hourly().to_numpy()
    occ = wk * (1 / (1 + np.exp(-(h - 7.3) * 3))) * (1 / (1 + np.exp((h - 17.5) * 3)))
    lunch = 1 - 0.15 * _g(h, 12.3, 0.5)
    cool = np.clip(t - 22, 0, None) * 0.05 * occ
    shape = (0.22 + occ * lunch + cool) * _noise(rng, len(idx), 0.10)
    return _scale_to_wh(shape, annual_kwh)


def bakery(rng: np.random.Generator, annual_kwh: float) -> np.ndarray:
    idx = _grid()
    h = (idx.hour + idx.minute / 60).to_numpy(dtype=float)
    dow = idx.day_of_week.to_numpy()
    open_day = np.where(dow == 6, 0.5, 1.0)
    ovens = np.where((h >= 2.5) & (h < 11), 1.0, 0.0) * (0.7 + 0.3 * _g(h, 5.5, 2.0)) * open_day
    shop = _g(h, 9.5, 3.5) * 0.25 * open_day
    shape = (0.30 + 1.6 * ovens + shop) * _noise(rng, len(idx), 0.15)
    return _scale_to_wh(shape, annual_kwh)


def shop(rng: np.random.Generator, annual_kwh: float) -> np.ndarray:
    idx = _grid()
    h = (idx.hour + idx.minute / 60).to_numpy(dtype=float)
    dow = idx.day_of_week.to_numpy()
    open_ = np.where(dow == 6, 0.0, 1.0) * (h >= 8) * (h < np.where(dow == 5, 17, 19))
    shape = (0.45 + 0.9 * open_) * _noise(rng, len(idx), 0.10)  # refrigeration keeps a high base load
    return _scale_to_wh(shape, annual_kwh)


def workshop(rng: np.random.Generator, annual_kwh: float) -> np.ndarray:
    idx = _grid()
    h = (idx.hour + idx.minute / 60).to_numpy(dtype=float)
    wk = (idx.day_of_week.to_numpy() < 5).astype(float)
    on = wk * (((h >= 7) & (h < 12)) | ((h >= 13) & (h < 17))).astype(float)
    bursts = np.zeros(len(idx))
    for d in range(365):
        for _ in range(rng.poisson(3)):
            s = d * SLOTS_PER_DAY + int(rng.uniform(7, 16) * 4)
            bursts[s : s + int(rng.integers(1, 5))] += rng.uniform(1.0, 2.5)
    shape = (0.10 + 0.5 * on + bursts * on) * _noise(rng, len(idx), 0.12)
    return _scale_to_wh(shape, annual_kwh)
