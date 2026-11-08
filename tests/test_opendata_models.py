import numpy as np
import pytest
from pydantic import ValidationError

from leg_sim import opendata
from leg_sim.models import Building, Community, LegParams, Member, Plant


def test_pv_year_shape_and_night_is_dark():
    s = opendata.pv_quarter_hourly("south", 1.0)
    assert len(s) == 365 * 96 and s.dtype == np.int64
    night = s[(s.index.hour <= 2) | (s.index.hour >= 22)]
    assert night.sum() == 0


@pytest.mark.parametrize("orientation,lo,hi", [("south", 1000, 1250), ("east", 850, 1100), ("west", 850, 1100)])
def test_pv_specific_yield_is_plausible_for_zurich(orientation, lo, hi):
    kwh = opendata.pv_quarter_hourly(orientation, 1.0).sum() / 1000
    assert lo < kwh < hi


def test_pv_scales_linearly_with_kwp():
    a = opendata.pv_quarter_hourly("south", 10).sum()
    b = opendata.pv_quarter_hourly("south", 20).sum()
    assert b == pytest.approx(2 * a, rel=1e-3)


def test_pv_hourly_energy_is_preserved_by_quarter_split():
    raw = opendata.load_pv_year()["p_south_w_per_kwp"].sum() * 1.0  # Wh per kWp
    assert opendata.pv_quarter_hourly("south", 1.0).sum() == pytest.approx(raw, abs=2)


def test_pv_summer_beats_winter():
    s = opendata.pv_quarter_hourly("south", 1.0)
    m = s.groupby(s.index.month).sum()
    assert m[6] > 3 * m[12]


def test_pv_rejects_bad_input():
    with pytest.raises(ValueError):
        opendata.pv_quarter_hourly("north", 1.0)
    with pytest.raises(ValueError):
        opendata.pv_quarter_hourly("south", -1.0)


def test_temperature_series_plausible():
    t = opendata.temperature_quarter_hourly()
    assert len(t) == 35040 and -15 < t.min() < 0 and 25 < t.max() < 40


def test_tariff_components_never_exceed_total_and_fixed_part_is_small():
    t = opendata.load_tariffs()
    s = t.grid_rp + t.energy_rp + t.charge_rp + t.aid_rp
    assert (s <= t.total_rp + 0.01).all()
    assert ((t.total_rp - s) < 4).all()  # fixed/metering part of the ElCom total, Rp per kWh


def test_fixed_charge_is_derived_from_the_residual():
    x = opendata.utility_tariff(486, "H4")
    row = opendata.load_tariffs().query("operator_id == 486 and category == 'H4'").iloc[0]
    resid = row.total_rp - row.grid_rp - row.energy_rp - row.charge_rp - row.aid_rp
    assert x["fixed_chf_month"] == pytest.approx(resid / 100 * 4500 / 12)


def test_tariff_lookup_values():
    x = opendata.utility_tariff(486, "H4")
    assert x["energy"] == pytest.approx(0.12112) and x["aid"] == pytest.approx(0.023)


def test_tariff_lookup_unknown_raises():
    with pytest.raises(KeyError):
        opendata.utility_tariff(1, "H4")


def _base():
    return dict(
        buildings=[Building(id="B1", transformer_station="T")],
        members=[Member(id="a", kind="household", building="B1")],
        plants=[],
    )


def test_community_rejects_duplicate_member_ids():
    kw = _base()
    kw["members"] = kw["members"] * 2
    with pytest.raises(ValidationError, match="duplicate"):
        Community(**kw)


def test_community_rejects_unknown_building():
    kw = _base()
    kw["members"] = [Member(id="a", kind="household", building="B9")]
    with pytest.raises(ValidationError, match="unknown building"):
        Community(**kw)


def test_plant_owner_must_be_a_producer():
    kw = _base()
    kw["plants"] = [Plant(id="p", building="B1", owner="a", kwp=5)]
    with pytest.raises(ValidationError, match="not a producer"):
        Community(**kw)


def test_member_id_must_be_filename_safe():
    with pytest.raises(ValidationError):
        Member(id="../x", kind="household", building="B1")


def test_community_needs_a_consumer():
    with pytest.raises(ValidationError, match="at least one consumer"):
        Community(buildings=[Building(id="B1", transformer_station="T")], members=[], plants=[])


def test_params_reject_out_of_range():
    with pytest.raises(ValidationError):
        LegParams(discount_single_ts=1.5)
