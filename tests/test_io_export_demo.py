import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from leg_sim import community as comm
from leg_sim import profiles
from leg_sim.billing import compute_bills, run_allocation
from leg_sim.cli import app
from leg_sim.demo import run_demo
from leg_sim.export import invoice_markdown, meter_data_xml, parse_meter_data, write_meter_data
from leg_sim.models import LegParams


def test_demo_community_structure():
    c = comm.demo_community()
    assert len(c.consumers) == 34 and len(c.producers) == 3
    assert sum(p.kwp for p in c.plants) == 110
    assert c.n_transformer_stations == 1
    assert comm.demo_community(single_transformer_station=False).n_transformer_stations == 2


def test_profile_hits_target_energy_and_is_integer():
    rng = np.random.default_rng(3)
    for fn in (profiles.common_area, profiles.office, profiles.bakery, profiles.shop, profiles.workshop):
        p = fn(rng, 12000.0)
        assert p.dtype == np.int64 and len(p) == 35040 and (p >= 0).all()
        assert p.sum() / 1000 == pytest.approx(12000.0, abs=0.5)


def test_household_profile_with_ev_and_reproducible():
    a = profiles.household(np.random.default_rng(5), 4500, 0.5, ev_kwh=2000)
    b = profiles.household(np.random.default_rng(5), 4500, 0.5, ev_kwh=2000)
    c = profiles.household(np.random.default_rng(6), 4500, 0.5, ev_kwh=2000)
    assert (a == b).all() and not (a == c).all()
    assert a.sum() / 1000 == pytest.approx(4500, rel=0.01)


def test_heat_pump_follows_temperature():
    p = pd.Series(profiles.central_heat_pump(np.random.default_rng(1), 10000.0))
    idx = pd.date_range("2023-01-01", periods=len(p), freq="15min")
    m = p.groupby(idx.month).sum()
    assert m[1] > 4 * m[7]


def test_office_is_busier_on_weekdays_in_daytime():
    p = pd.Series(profiles.office(np.random.default_rng(2), 18000.0), index=pd.date_range("2023-01-01", periods=35040, freq="15min"))
    wk_noon = p[(p.index.day_of_week < 5) & (p.index.hour == 11)].mean()
    sun_noon = p[(p.index.day_of_week == 6) & (p.index.hour == 11)].mean()
    assert wk_noon > 2 * sun_noon


def test_profile_csv_roundtrip(tmp_path, tiny):
    _, loads, _ = tiny
    f = tmp_path / "l.csv"
    comm.write_profiles(loads, f)
    back = comm.read_profiles(f, ["a", "b", "shop"])
    assert (back.to_numpy() == loads.to_numpy()).all() and back.index.equals(loads.index)


def test_read_profiles_rejects_gap(tmp_path, tiny):
    _, loads, _ = tiny
    f = tmp_path / "gap.csv"
    comm.write_profiles(loads.drop(loads.index[10]), f)
    with pytest.raises(ValueError, match="gap-free"):
        comm.read_profiles(f)


def test_read_profiles_rejects_negative_and_missing_column(tmp_path, tiny):
    _, loads, _ = tiny
    bad = loads.copy()
    bad.iloc[3, 0] = -5
    f = tmp_path / "neg.csv"
    comm.write_profiles(bad, f)
    with pytest.raises(ValueError, match="negative"):
        comm.read_profiles(f)
    g = tmp_path / "ok.csv"
    comm.write_profiles(loads, g)
    with pytest.raises(ValueError, match="missing columns"):
        comm.read_profiles(g, ["a", "zzz"])


def test_read_profiles_rejects_garbage(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("timestamp,a\nnot-a-date,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="timestamps"):
        comm.read_profiles(f)
    h = tmp_path / "y.csv"
    h.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="timestamp"):
        comm.read_profiles(h)


def test_meter_data_xml_roundtrip_matches_allocation(tiny):
    c, loads, plants = tiny
    res = run_allocation(c, loads, plants, "dynamic")
    msg = meter_data_xml("a", res.alloc.index[0], res.alloc["a"])
    d = parse_meter_data(msg)
    assert d["metering_point"] == "a" and d["resolution"] == "PT15M" and d["unit"] == "KWH"
    assert d["sequence"] == list(range(1, 97))
    assert sum(d["volumes_kwh"]) == pytest.approx(res.alloc["a"].sum() / 1000, abs=0.0005 * 96)
    assert d["start"] == "2023-03-01T00:00:00" and d["end"] == "2023-03-02T00:00:00"


def test_meter_data_xml_rejects_empty_and_negative():
    with pytest.raises(ValueError):
        meter_data_xml("a", pd.Timestamp("2023-01-01"), pd.Series([], dtype="int64"))
    with pytest.raises(ValueError):
        meter_data_xml("a", pd.Timestamp("2023-01-01"), pd.Series([-1, 2]))


def test_write_meter_data_one_file_per_consumer_and_bad_month(tmp_path, tiny):
    c, loads, plants = tiny
    res = run_allocation(c, loads, plants, "static")
    paths = write_meter_data(res, tmp_path, "2023-03")
    assert len(paths) == 3 and all(p.exists() for p in paths)
    with pytest.raises(ValueError, match="outside"):
        write_meter_data(res, tmp_path, "2023-07")
    with pytest.raises(ValueError, match="YYYY-MM"):
        write_meter_data(res, tmp_path, "july")


def test_invoice_markdown_contains_totals(tiny):
    c, loads, plants = tiny
    res = run_allocation(c, loads, plants, "dynamic")
    b, _ = compute_bills(c, res, LegParams())
    md = invoice_markdown(b.iloc[0], c.name, 0.4, "dynamic")
    assert f"{b.iloc[0]['bill_leg_chf']:.2f}" in md and "SIMULATED" in md and "40%" in md


def test_cli_simulate_end_to_end_and_errors(tmp_path, tiny):
    c, loads, plants = tiny
    (tmp_path / "c.json").write_text(c.model_dump_json(), encoding="utf-8")
    comm.write_profiles(loads, tmp_path / "l.csv")
    comm.write_profiles(plants, tmp_path / "p.csv")
    r = CliRunner()
    args = ["simulate", "--community", str(tmp_path / "c.json"), "--loads", str(tmp_path / "l.csv"),
            "--pv", str(tmp_path / "p.csv"), "--out", str(tmp_path / "o")]  # fmt: skip
    res = r.invoke(app, args + ["--method", "dynamic"])
    assert res.exit_code == 0, res.output
    assert "balance error 0 Wh" in res.output
    assert (tmp_path / "o" / "bills_monthly.csv").exists() and any((tmp_path / "o" / "invoices").glob("*.md"))
    assert r.invoke(app, args + ["--method", "bogus"]).exit_code != 0
    (tmp_path / "p.csv").write_text("timestamp,pv1\nxx,1\n", encoding="utf-8")
    bad = r.invoke(app, args)
    assert bad.exit_code == 2


def test_cli_export_meter_data(tmp_path, tiny):
    c, loads, plants = tiny
    (tmp_path / "c.json").write_text(c.model_dump_json(), encoding="utf-8")
    comm.write_profiles(loads, tmp_path / "l.csv")
    comm.write_profiles(plants, tmp_path / "p.csv")
    args = ["export-meter-data", "--community", str(tmp_path / "c.json"), "--loads", str(tmp_path / "l.csv")]
    args += ["--pv", str(tmp_path / "p.csv"), "--month", "2023-03", "--out", str(tmp_path / "x")]
    res = CliRunner().invoke(app, args)
    assert res.exit_code == 0 and len(list((tmp_path / "x").glob("*.xml"))) == 3


@pytest.fixture(scope="module")
def demo_summary(tmp_path_factory):
    return run_demo(tmp_path_factory.mktemp("demo"), None, seed=7, write_files=False)


def test_demo_balance_exact_for_all_methods(demo_summary):
    assert all(m["balance_error_wh"] == 0 for m in demo_summary["methods"].values())


def test_demo_dynamic_beats_static_and_everyone_saves(demo_summary):
    m = demo_summary["methods"]
    assert m["dynamic"]["self_consumption_pct"] > m["static"]["self_consumption_pct"] + 3
    assert m["fair"]["self_consumption_pct"] == pytest.approx(m["dynamic"]["self_consumption_pct"], abs=0.01)
    assert all(m[k]["saving_chf_min"] > 0 for k in m)


def test_demo_scenarios_ordered_by_discount(demo_summary):
    s = {x["name"]: x["total_saving_chf"] for x in demo_summary["scenarios"]}
    lo, mid, hi = (s["LEG, no grid-fee reduction (0%)"], s["LEG, two transformer stations (20%)"],
                   s["LEG, one transformer station (40%)"])  # fmt: skip
    assert lo < mid < hi


def test_demo_is_reproducible(tmp_path):
    a = run_demo(tmp_path / "a", None, seed=11, write_files=False)
    b = run_demo(tmp_path / "b", None, seed=11, write_files=False)
    assert a["methods"] == b["methods"]
