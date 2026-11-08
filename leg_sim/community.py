"""Demo community (SIMULATED) and CSV input/output of meter profiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from leg_sim import profiles
from leg_sim.models import Building, Community, Member, Plant
from leg_sim.opendata import pv_quarter_hourly

SLOTS = 365 * 96
YEAR = 2023


def demo_community(single_transformer_station: bool = True) -> Community:
    """4 buildings, 34 consumers (26 households, 4 businesses, 4 common-area meters), 3 PV owners, 110 kWp."""
    ts = ["TS1", "TS1", "TS1", "TS1" if single_transformer_station else "TS2"]
    buildings = [Building(id=f"B{i + 1}", transformer_station=t) for i, t in enumerate(ts)]
    members: list[Member] = []
    hh_per_building = [8, 7, 6, 5]
    n = 0
    for b, cnt in zip(buildings, hh_per_building, strict=True):
        for _ in range(cnt):
            n += 1
            members.append(Member(id=f"hh{n:02d}", kind="household", building=b.id, category="H4"))
    members += [
        Member(id="bakery", kind="bakery", building="B2", category="C2"),
        Member(id="shop", kind="shop", building="B3", category="C2"),
        Member(id="office", kind="office", building="B3", category="C2"),
        Member(id="workshop", kind="workshop", building="B4", category="C2"),
    ]
    members += [Member(id=f"common_b{i}", kind="common_area", building=f"B{i}", category="C2") for i in (1, 2, 3, 4)]
    members += [Member(id=f"pv_owner_b{i}", kind="pv_owner", role="producer", building=f"B{i}") for i in (1, 2, 3)]
    plants = [
        Plant(id="pv_b1_south", building="B1", owner="pv_owner_b1", orientation="south", kwp=60),
        Plant(id="pv_b2_east", building="B2", owner="pv_owner_b2", orientation="east", kwp=15),
        Plant(id="pv_b2_west", building="B2", owner="pv_owner_b2", orientation="west", kwp=15),
        Plant(id="pv_b3_south", building="B3", owner="pv_owner_b3", orientation="south", kwp=20),
    ]
    return Community(name="demo-leg-34+3", buildings=buildings, members=members, plants=plants)


def simulate_loads(community: Community, seed: int = 7) -> pd.DataFrame:
    """Simulated 15-min consumption (integer Wh) for every consumer; also fills ``annual_kwh`` on members."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{YEAR}-01-01", periods=SLOTS, freq="15min")
    cols: dict[str, np.ndarray] = {}
    hh_types = [  # (annual kWh, share of day at home, EV kWh)
        (1700, 0.5, 0), (2300, 0.3, 0), (2900, 0.9, 0), (3500, 0.4, 0), (4500, 0.6, 0), (3300, 1.0, 0),
        (5600, 0.5, 2500), (2700, 0.3, 0), (4200, 0.7, 0), (3000, 0.4, 2200),
    ]  # fmt: skip
    for m in community.consumers:
        if m.kind == "household":
            kwh, home, ev = hh_types[int(rng.integers(0, len(hh_types)))]
            kwh = float(kwh * rng.uniform(0.9, 1.1))
            cols[m.id] = profiles.household(rng, kwh, home, ev_kwh=float(ev))
        elif m.kind == "common_area":
            hp = m.building in ("B1", "B3")
            base = profiles.common_area(rng, 2600 * rng.uniform(0.9, 1.1))
            cols[m.id] = base + (profiles.central_heat_pump(rng, 14500 * rng.uniform(0.9, 1.1)) if hp else 0)
        elif m.kind == "bakery":
            cols[m.id] = profiles.bakery(rng, 28000)
        elif m.kind == "shop":
            cols[m.id] = profiles.shop(rng, 16000)
        elif m.kind == "office":
            cols[m.id] = profiles.office(rng, 18000)
        elif m.kind == "workshop":
            cols[m.id] = profiles.workshop(rng, 12000)
        else:
            raise ValueError(f"no simulator for member kind {m.kind!r}")
    df = pd.DataFrame(cols, index=idx).astype("int64")
    for m in community.consumers:
        m.annual_kwh = float(df[m.id].sum()) / 1000.0
    return df


def simulate_production(community: Community) -> pd.DataFrame:
    """15-min PV energy (integer Wh) per plant from PVGIS hourly yield (see ``opendata.pv_quarter_hourly``)."""
    return pd.DataFrame({p.id: pv_quarter_hourly(p.orientation, p.kwp, YEAR) for p in community.plants})


def write_profiles(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out.index.name = "timestamp"
    out.to_csv(path, date_format="%Y-%m-%dT%H:%M:%S")


def read_profiles(path: Path, expected_columns: list[str] | None = None) -> pd.DataFrame:
    """Read a 15-minute Wh CSV (column ``timestamp`` + one column per meter). Validates grid, gaps and values."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # malformed file
        raise ValueError(f"cannot read profile CSV {path}: {exc}") from exc
    if "timestamp" not in df.columns:
        raise ValueError("profile CSV needs a 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("unparseable timestamps in profile CSV")
    df = df.set_index("timestamp").sort_index()
    if df.index.has_duplicates:
        raise ValueError("duplicate timestamps in profile CSV")
    if len(df) > 1 and not (df.index.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all():
        raise ValueError("profile CSV must be a gap-free 15-minute series")
    if df.isna().any().any():
        raise ValueError("missing values in profile CSV")
    if (df < 0).any().any():
        raise ValueError("negative energy values in profile CSV")
    if expected_columns is not None:
        missing = [c for c in expected_columns if c not in df.columns]
        if missing:
            raise ValueError(f"profile CSV is missing columns: {missing}")
        df = df[expected_columns]
    return df.round().astype("int64")
