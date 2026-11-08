"""End-to-end demo: simulate a 34+3 member community for 2023, run 3 methods, bill, export, plot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from leg_sim import community as comm
from leg_sim.allocation import METHODS, static_shares
from leg_sim.billing import community_metrics, compute_bills, internal_price, run_allocation
from leg_sim.export import write_invoices, write_meter_data
from leg_sim.models import LegParams
from leg_sim.opendata import ELCOM_OPERATORS, utility_tariff

SOURCE_LINE = "Simulated community, 2023 | PV: PVGIS-SARAH3 (EU JRC) | tariffs: ElCom 2026 (opendata.swiss)"
WEEK_START = "2023-06-19"


def _hh_stats(bills: pd.DataFrame) -> dict:
    hh = bills[bills["kind"] == "household"].groupby("member")[["saving_chf", "bill_no_leg_chf"]].sum()
    s = hh["saving_chf"]
    return {
        "n": int(len(s)),
        "median_chf": float(s.median()),
        "p10_chf": float(s.quantile(0.1)),
        "p90_chf": float(s.quantile(0.9)),
        "min_chf": float(s.min()),
        "max_chf": float(s.max()),
        "median_pct_of_bill": float((s / hh["bill_no_leg_chf"] * 100).median()),
        "values": s.sort_values(ascending=False).round(2).tolist(),
    }


def run_demo(out: Path, docs: Path | None = None, seed: int = 7, write_files: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    c = comm.demo_community()
    loads = comm.simulate_loads(c, seed=seed)
    plants = comm.simulate_production(c)
    shares = static_shares(np.array([m.annual_kwh for m in c.consumers]))
    base = LegParams()

    results, metrics, hh_arr, hh = {}, {}, {}, {}
    bills_by_method = {}
    for method in METHODS:
        r = run_allocation(c, loads, plants, method, shares)
        b, p = compute_bills(c, r, base)
        results[method], bills_by_method[method] = r, b
        metrics[method] = community_metrics(r, b, p)
        hh[method] = _hh_stats(b)
        hh_arr[method] = np.array(hh[method]["values"])

    dyn = results["dynamic"]
    sweep = []
    for op in ELCOM_OPERATORS:
        prm = base.model_copy(update={"operator_id": op})
        b, p = compute_bills(c, dyn, prm)
        h = _hh_stats(b)
        sweep.append(
            dict(
                operator_id=op, bfs_municipality=utility_tariff(op, "H4")["bfs_municipality"],
                household_tariff_chf_kwh=sum(utility_tariff(op, "H4")[k] for k in ("grid", "energy", "charge", "aid")),
                internal_price_chf_kwh=internal_price(prm), hh_median_saving_chf=h["median_chf"], hh_p90_saving_chf=h["p90_chf"],
                producer_gain_chf=float(p["gain_chf"].sum()),
            )  # fmt: skip
        )

    def scen(name: str, community, params: LegParams) -> dict:
        b, p = compute_bills(community, dyn, params)
        h = _hh_stats(b)
        return dict(
            name=name, grid_discount=params.grid_discount(community), hh_median_saving_chf=h["median_chf"],
            total_saving_chf=float(b["saving_chf"].sum()), producer_gain_chf=float(p["gain_chf"].sum()),
        )  # fmt: skip

    c2 = comm.demo_community(single_transformer_station=False)
    scenarios = [
        scen("LEG, one transformer station (40%)", c, base),
        scen("LEG, two transformer stations (20%)", c2, base),
        scen("LEG, 40% on whole grid fee (share 1.0)", c, base.model_copy(update={"discountable_share": 1.0})),
        scen("LEG, 40%, billing fee 1 CHF/month", c, base.model_copy(update={"admin_fee_chf_month": 1.0})),
        scen("LEG, no grid-fee reduction (0%)", c, base.model_copy(update={"discount_override": 0.0})),
        scen("self-consumption group (ZEV), no grid fee", c, base.model_copy(update={"scheme": "zev"})),
    ]

    summary = dict(
        community=dict(
            consumers=len(c.consumers), producers=len(c.producers), buildings=len(c.buildings),
            pv_kwp=float(sum(p.kwp for p in c.plants)), transformer_stations=c.n_transformer_stations,
        ),
        params=base.model_dump(), grid_discount=base.grid_discount(c), internal_price_chf_kwh=internal_price(base),
        methods=metrics, household_stats={k: {kk: vv for kk, vv in v.items() if kk != "values"} for k, v in hh.items()},
        operator_sweep=sweep, scenarios=scenarios, seed=seed,
    )  # fmt: skip

    if write_files:
        (out / "data").mkdir(exist_ok=True)
        comm.write_profiles(loads, out / "data" / "loads_wh_15min.csv")
        comm.write_profiles(plants, out / "data" / "pv_wh_15min.csv")
        (out / "community.json").write_text(c.model_dump_json(indent=2), encoding="utf-8")
        (out / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for m in METHODS:
            bills_by_method[m].to_csv(out / f"bills_monthly_{m}.csv", index=False, float_format="%.4f")
        b, p = compute_bills(c, results["fair"], base)
        p.to_csv(out / "producer_revenue_monthly_fair.csv", index=False, float_format="%.4f")
        write_invoices(b, out / "invoices", c.name, base.grid_discount(c), "fair")
        write_meter_data(results["fair"], out / "meter_data_xml", "2023-07")
        if docs is not None:
            docs.mkdir(parents=True, exist_ok=True)
            from leg_sim import plots

            plots.main_figure(results, metrics, hh_arr, docs / "figure.png", WEEK_START, SOURCE_LINE)
            plots.post_figure(
                dyn, hh_arr["dynamic"], docs / "post-figure.png", WEEK_START, SOURCE_LINE,
                metrics["dynamic"]["self_consumption_pct"],
            )  # fmt: skip
    return summary


def format_summary(s: dict) -> str:
    """Console/markdown summary of the headline numbers."""
    lines = [
        f"Community: {s['community']['consumers']} consumers + {s['community']['producers']} PV owners, "
        f"{s['community']['pv_kwp']:.0f} kWp, grid-fee reduction {s['grid_discount']:.0%}, "
        f"community price {s['internal_price_chf_kwh'] * 100:.1f} Rp/kWh",
        "",
        "| method | self-consumption % | self-sufficiency % | hh median CHF/yr | hh p10 | hh p90 | Jain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m, v in s["methods"].items():
        h = s["household_stats"][m]
        lines.append(
            f"| {m} | {v['self_consumption_pct']:.1f} | {v['self_sufficiency_pct']:.1f} | {h['median_chf']:.0f} | "
            f"{h['p10_chf']:.0f} | {h['p90_chf']:.0f} | {v['jain_self_sufficiency']:.3f} |"
        )
    return "\n".join(lines)
