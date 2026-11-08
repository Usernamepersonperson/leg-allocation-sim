import numpy as np
import pandas as pd
import pytest

from leg_sim.models import Building, Community, Member, Plant


@pytest.fixture
def tiny():
    """3 consumers + 1 producer, one day of 15-minute intervals, hand-checkable."""
    idx = pd.date_range("2023-03-01", periods=96, freq="15min")
    c = Community(
        buildings=[Building(id="B1", transformer_station="TS1")],
        members=[
            Member(id="a", kind="household", building="B1", annual_kwh=1000),
            Member(id="b", kind="household", building="B1", annual_kwh=3000),
            Member(id="shop", kind="shop", building="B1", category="C2", annual_kwh=6000),
            Member(id="owner", kind="pv_owner", role="producer", building="B1"),
        ],
        plants=[Plant(id="pv1", building="B1", owner="owner", kwp=10)],
    )
    rng = np.random.default_rng(1)
    loads = pd.DataFrame(rng.integers(0, 400, size=(96, 3)), index=idx, columns=["a", "b", "shop"]).astype("int64")
    day = np.clip(np.sin((np.arange(96) - 28) / 40 * np.pi), 0, None)
    plants = pd.DataFrame({"pv1": (day * 1500).astype("int64")}, index=idx)
    return c, loads, plants


@pytest.fixture
def rand_case():
    rng = np.random.default_rng(42)
    loads = rng.integers(0, 600, size=(500, 12)).astype("int64")
    loads[rng.random(loads.shape) < 0.2] = 0
    prod = rng.integers(0, 4000, size=500).astype("int64")
    shares = rng.random(12) + 0.1
    return loads, prod, shares / shares.sum()
