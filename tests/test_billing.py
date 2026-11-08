import numpy as np
import pandas as pd
import pytest

from leg_sim.billing import (
    AllocationResult,
    community_metrics,
    compute_bills,
    internal_price,
    jain_index,
    run_allocation,
)
from leg_sim.models import Building, Community, LegParams, Member, Plant
from leg_sim.opendata import utility_tariff


def _manual(tiny, alloc_wh_per_quarter=500, load_wh=1000, prod_wh=2000):
    c, _, _ = tiny
    idx = pd.date_range("2023-03-01", periods=96, freq="15min")
    cons = [m.id for m in c.consumers]
    loads = pd.DataFrame(0, index=idx, columns=cons, dtype="int64")
    alloc = loads.copy()
    loads["a"] = load_wh
    alloc["a"] = alloc_wh_per_quarter
    production = pd.Series(prod_wh, index=idx, dtype="int64")
    plants = pd.DataFrame({"pv1": production})
    return AllocationResult("manual", loads, alloc, production, production - alloc.sum(axis=1), plants)


def test_bill_hand_calculation(tiny):
    c = tiny[0]
    res = _manual(tiny)
    p = LegParams(operator_id=486)
    bills, _ = compute_bills(c, res, p)
    row = bills[bills.member == "a"].iloc[0]
    t = utility_tariff(486, "H4")
    full = t["grid"] + t["energy"] + t["charge"] + t["aid"]
    p_int = p.feed_in_chf_kwh + 0.5 * (t["energy"] - p.feed_in_chf_kwh)
    int_rate = p_int + t["grid"] * (1 - 0.4 * 0.9) + t["charge"] + t["aid"]
    assert row.kwh_total == pytest.approx(96.0)
    assert row.kwh_community == pytest.approx(48.0)
    fixed = t["fixed_chf_month"]
    assert row.bill_no_leg_chf == pytest.approx(96 * full + fixed)
    assert row.bill_leg_chf == pytest.approx(48 * full + 48 * int_rate + fixed)
    assert row.saving_chf == pytest.approx(48 * (full - int_rate))


def test_zero_saving_when_no_discount_and_price_equals_tariff(tiny):
    c = tiny[0]
    t = utility_tariff(486, "H4")
    p = LegParams(operator_id=486, discount_override=0.0, internal_price_chf_kwh=t["energy"])
    bills, _ = compute_bills(c, _manual(tiny), p)
    assert bills[bills.member == "a"].saving_chf.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_admin_fee_reduces_saving_by_the_fee(tiny):
    c = tiny[0]
    res = _manual(tiny)
    b0, _ = compute_bills(c, res, LegParams())
    b1, _ = compute_bills(c, res, LegParams(admin_fee_chf_month=3.0))
    d = b0[b0.member == "a"].saving_chf.iloc[0] - b1[b1.member == "a"].saving_chf.iloc[0]
    assert d == pytest.approx(3.0)


def test_members_without_consumption_pay_only_the_fee(tiny):
    c = tiny[0]
    b, _ = compute_bills(c, _manual(tiny), LegParams(admin_fee_chf_month=2.0))
    row = b[b.member == "b"].iloc[0]
    fixed = utility_tariff(486, "H4")["fixed_chf_month"]
    assert row.bill_leg_chf == pytest.approx(2.0 + fixed) and row.bill_no_leg_chf == pytest.approx(fixed)


def test_community_kwh_reconciles_with_allocation(tiny):
    c, loads, plants = tiny
    res = run_allocation(c, loads, plants, "dynamic")
    b, _ = compute_bills(c, res, LegParams())
    assert b.kwh_community.sum() == pytest.approx(res.alloc.to_numpy().sum() / 1000)
    assert b.kwh_total.sum() == pytest.approx(loads.to_numpy().sum() / 1000)
    assert res.balance_error_wh() == 0


def test_producer_gain_equals_sold_times_price_spread(tiny):
    c, loads, plants = tiny
    res = run_allocation(c, loads, plants, "dynamic")
    p = LegParams()
    _, prod = compute_bills(c, res, p)
    sold_kwh = res.alloc.to_numpy().sum() / 1000
    assert prod.gain_chf.sum() == pytest.approx(sold_kwh * (internal_price(p) - p.feed_in_chf_kwh))
    assert prod.kwh_produced.sum() == pytest.approx(plants.to_numpy().sum() / 1000)


def test_two_plants_share_revenue_by_production(tiny):
    c, loads, plants = tiny
    c2 = Community(
        buildings=c.buildings,
        members=c.members + [Member(id="owner2", kind="pv_owner", role="producer", building="B1")],
        plants=c.plants + [Plant(id="pv2", building="B1", owner="owner2", kwp=10)],
    )
    plants2 = plants.assign(pv2=plants["pv1"] * 3)
    res = run_allocation(c2, loads, plants2, "dynamic")
    _, prod = compute_bills(c2, res, LegParams())
    g = prod.groupby("member").gain_chf.sum()
    assert g["owner2"] == pytest.approx(3 * g["owner"], rel=1e-6)


def test_grid_discount_rules(tiny):
    c = tiny[0]
    assert LegParams().grid_discount(c) == 0.40
    assert LegParams(discount_override=0.1).grid_discount(c) == 0.1
    assert LegParams(scheme="zev").grid_discount(c) == 1.0


def test_multiple_transformer_stations_use_lower_discount(tiny):
    c = tiny[0]
    c2 = Community(
        buildings=[Building(id="B1", transformer_station="TS1"), Building(id="B2", transformer_station="TS2")],
        members=c.members + [Member(id="d", kind="household", building="B2")],
        plants=c.plants,
    )
    assert c2.n_transformer_stations == 2
    assert LegParams().grid_discount(c2) == 0.20


def test_internal_price_between_feed_in_and_tariff():
    p = LegParams(operator_id=486)
    t = utility_tariff(486, "H4")["energy"]
    assert p.feed_in_chf_kwh < internal_price(p) < t
    assert internal_price(LegParams(internal_price_chf_kwh=0.07)) == 0.07


def test_internal_price_never_below_feed_in_when_tariff_is_lower():
    p = LegParams(operator_id=565)  # energy tariff below the assumed feed-in remuneration
    assert internal_price(p) == p.feed_in_chf_kwh


def test_zev_saves_more_than_leg(tiny):
    c, loads, plants = tiny
    res = run_allocation(c, loads, plants, "dynamic")
    leg, _ = compute_bills(c, res, LegParams())
    zev, _ = compute_bills(c, res, LegParams(scheme="zev"))
    assert zev.saving_chf.sum() > leg.saving_chf.sum() > 0


def test_jain_index_extremes():
    assert jain_index(np.array([2, 2, 2, 2])) == pytest.approx(1.0)
    assert jain_index(np.array([1, 0, 0, 0])) == pytest.approx(0.25)
    assert jain_index(np.array([])) == 1.0


def test_metrics_rates_are_consistent(tiny):
    c, loads, plants = tiny
    res = run_allocation(c, loads, plants, "fair")
    b, p = compute_bills(c, res, LegParams())
    m = community_metrics(res, b, p)
    assert 0 <= m["self_sufficiency_pct"] <= 100 and 0 <= m["self_consumption_pct"] <= 100
    assert m["community_kwh"] + m["export_kwh"] == pytest.approx(m["production_kwh"])
    assert m["balance_error_wh"] == 0


def test_run_allocation_index_mismatch_raises(tiny):
    c, loads, plants = tiny
    with pytest.raises(ValueError, match="same 15-minute index"):
        run_allocation(c, loads, plants.iloc[:-1], "static")


def test_run_allocation_unknown_method_raises(tiny):
    c, loads, plants = tiny
    with pytest.raises(ValueError):
        run_allocation(c, loads, plants, "nope")
