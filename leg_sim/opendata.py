"""Open-data access: PVGIS PV yield (EU JRC) and ElCom 2026 electricity tariffs (LINDAS).

The committed CSVs in ``leg_sim/data`` are compact extracts. ``fetch_pvgis`` / ``fetch_elcom`` rebuild them.
"""

from __future__ import annotations

import io
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
PVGIS_FILE = DATA_DIR / "pvgis_zurich_2023.csv"
ELCOM_FILE = DATA_DIR / "elcom_tariffs_2026.csv"

PVGIS_URL = (
    "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc?lat=47.37&lon=8.54&startyear=2023&endyear=2023"
    "&pvcalculation=1&peakpower=1&loss=14&outputformat=csv&angle={angle}&aspect={aspect}"
)
# name -> (slope deg, azimuth deg; 0 = south, -90 = east, 90 = west)
PVGIS_ORIENTATIONS = {"south": (30, 0), "east": (15, -90), "west": (15, 90)}

LINDAS_URL = "https://ld.admin.ch/query"
# ElCom operator ids (public identifiers) used for the demo; operator names are deliberately not stored.
ELCOM_OPERATORS = [565, 486, 36, 609, 692, 624, 37]
ELCOM_CATEGORIES = ["H4", "C2"]
# ElCom reference consumption per category (kWh/year); turns the fixed part of the ElCom total into CHF/month
REF_KWH = {"H4": 4500, "C2": 30000}
_NS = "https://energy.ld.admin.ch/elcom/electricityprice"

_SPARQL = """
PREFIX e: <{ns}/dimension/>
SELECT ?op ?muni ?cat ?grid ?energy ?charge ?aid ?total WHERE {{ GRAPH <https://lindas.admin.ch/elcom/electricityprice> {{
 ?o a <https://cube.link/Observation> ; e:period "2026"^^<http://www.w3.org/2001/XMLSchema#gYear> ;
   e:operator ?op ; e:category ?cat ; e:product <{ns}/product/standard> ; e:municipality ?muni ;
   e:gridusage ?grid ; e:energy ?energy ; e:charge ?charge ; e:aidfee ?aid ; e:total ?total .
 VALUES ?op {{ {ops} }}
 VALUES ?cat {{ {cats} }}
}} }}
"""


def _http_get(url: str, data: bytes | None = None, accept: str = "text/csv") -> str:
    req = urllib.request.Request(url, data=data, headers={"Accept": accept, "User-Agent": "leg-allocation-sim"})
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 - fixed https URLs
        return r.read().decode("utf-8")


def fetch_pvgis(out: Path = PVGIS_FILE) -> Path:
    """Download hourly PV yield per kWp for 3 orientations (PVGIS-SARAH3, 2023, Zurich) and cache a compact CSV."""
    cols: dict[str, pd.Series] = {}
    t2m = None
    for name, (angle, aspect) in PVGIS_ORIENTATIONS.items():
        lines = _http_get(PVGIS_URL.format(angle=angle, aspect=aspect)).splitlines()
        start = next(i for i, ln in enumerate(lines) if ln.startswith("time,"))
        end = next((i for i, ln in enumerate(lines) if i > start and not ln.strip()), len(lines))
        df = pd.read_csv(io.StringIO("\n".join(lines[start:end])))
        df["time"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M").dt.floor("h")
        df = df.set_index("time")
        cols[f"p_{name}_w_per_kwp"] = df["P"]
        if t2m is None:
            t2m = df["T2m"]
    res = pd.DataFrame(cols)
    res["t2m_c"] = t2m
    res.index.name = "time_utc"
    res.to_csv(out)
    return out


def fetch_elcom(out: Path = ELCOM_FILE) -> Path:
    """Download ElCom 2026 tariffs (Rp/kWh, standard product) for the demo operators from LINDAS."""
    ops = " ".join(f"<{_NS}/operator/{o}>" for o in ELCOM_OPERATORS)
    cats = " ".join(f"<{_NS}/category/{c}>" for c in ELCOM_CATEGORIES)
    q = _SPARQL.format(ns=_NS, ops=ops, cats=cats)
    txt = _http_get(LINDAS_URL, data=urllib.parse.urlencode({"query": q}).encode())
    raw = pd.read_csv(io.StringIO(txt))
    raw["operator_id"] = raw["op"].str.rsplit("/", n=1).str[-1].astype(int)
    raw["bfs_municipality"] = raw["muni"].str.rsplit("/", n=1).str[-1].astype(int)
    raw["category"] = raw["cat"].str.rsplit("/", n=1).str[-1]
    rows = []
    for op, g in raw[raw["category"] == "H4"].groupby("operator_id"):
        g = g.sort_values(["total", "bfs_municipality"]).reset_index(drop=True)
        muni = g.iloc[(len(g) - 1) // 2]["bfs_municipality"]  # median-priced municipality (household category)
        rows.extend(r for _, r in raw[(raw["operator_id"] == op) & (raw["bfs_municipality"] == muni)].iterrows())
    cols = ["operator_id", "bfs_municipality", "category", "grid", "energy", "charge", "aid", "total"]
    out_df = pd.DataFrame(rows)[cols].rename(
        columns={"grid": "grid_rp", "energy": "energy_rp", "charge": "charge_rp", "aid": "aid_rp", "total": "total_rp"}
    )
    out_df.sort_values(["operator_id", "category"]).to_csv(out, index=False)
    return out


@lru_cache(maxsize=1)
def load_pv_year() -> pd.DataFrame:
    """Hourly PV yield per kWp, columns p_<orientation>_w_per_kwp and t2m_c, UTC naive index."""
    return pd.read_csv(PVGIS_FILE, index_col=0, parse_dates=True)


def _round_preserving(x: np.ndarray) -> np.ndarray:
    """Round to integer Wh keeping the running total within 1 Wh (error diffusion)."""
    c = np.cumsum(x)
    return np.diff(np.round(c), prepend=0).astype(np.int64)


def pv_quarter_hourly(orientation: str, kwp: float, year: int = 2023) -> pd.Series:
    """15-minute PV energy (integer Wh per interval) in local standard time (UTC+1).

    Source is hourly: each hour's energy is split over its 4 quarters using a linear interpolation of the
    hourly power (energy-preserving per hour). Sub-hourly cloud variability is NOT represented.
    """
    pv = load_pv_year()
    col = f"p_{orientation}_w_per_kwp"
    if col not in pv:
        raise ValueError(f"unknown PV orientation {orientation!r}")
    if kwp < 0:
        raise ValueError("kwp must be >= 0")
    wh_hour = pv[col].to_numpy(dtype=float) * kwp  # W * 1 h = Wh
    wh_hour = np.roll(wh_hour, 1)  # UTC -> UTC+1 local standard time
    n = len(wh_hour)
    centres = np.arange(n) + 0.5
    q_mid = np.arange(n * 4) / 4 + 0.125
    p_q = np.clip(np.interp(q_mid, centres, wh_hour), 0, None).reshape(n, 4)
    hour_sum = p_q.sum(axis=1)
    scale = np.divide(wh_hour, hour_sum, out=np.zeros(n), where=hour_sum > 0)
    q = (p_q * scale[:, None]).reshape(-1)
    idx = pd.date_range(f"{year}-01-01", periods=n * 4, freq="15min")
    return pd.Series(_round_preserving(q), index=idx, name=f"pv_{orientation}_{kwp:g}kwp")


def temperature_quarter_hourly(year: int = 2023) -> pd.Series:
    """2 m air temperature (deg C) on the 15-minute grid, local standard time."""
    t = np.roll(load_pv_year()["t2m_c"].to_numpy(dtype=float), 1)
    n = len(t)
    v = np.interp(np.arange(n * 4) / 4 + 0.125, np.arange(n) + 0.5, t)
    return pd.Series(v, index=pd.date_range(f"{year}-01-01", periods=n * 4, freq="15min"), name="t2m_c")


@lru_cache(maxsize=1)
def load_tariffs() -> pd.DataFrame:
    return pd.read_csv(ELCOM_FILE)


def utility_tariff(operator_id: int, category: str) -> dict[str, float]:
    """ElCom 2026 tariff components in CHF/kWh for an operator and consumer category."""
    t = load_tariffs()
    row = t[(t.operator_id == operator_id) & (t.category == category)]
    if row.empty:
        raise KeyError(f"no ElCom tariff for operator {operator_id}, category {category}")
    r = row.iloc[0]
    ref_kwh = REF_KWH[category]
    fixed_rp = max(0.0, float(r.total_rp) - float(r.grid_rp) - float(r.energy_rp) - float(r.charge_rp) - float(r.aid_rp))
    return {
        "fixed_chf_month": fixed_rp / 100 * ref_kwh / 12,
        "grid": float(r.grid_rp) / 100,
        "energy": float(r.energy_rp) / 100,
        "charge": float(r.charge_rp) / 100,
        "aid": float(r.aid_rp) / 100,
        "bfs_municipality": int(r.bfs_municipality),
    }
