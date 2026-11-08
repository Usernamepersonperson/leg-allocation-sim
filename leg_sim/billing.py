"""Allocation run, monthly billing and community metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from leg_sim.allocation import allocate, static_shares
from leg_sim.models import Community, LegParams
from leg_sim.opendata import utility_tariff


@dataclass
class AllocationResult:
    method: str
    loads: pd.DataFrame  # consumers, Wh
    alloc: pd.DataFrame  # consumers, Wh from the community
    production: pd.Series  # community total, Wh
    export: pd.Series  # Wh not used inside the community
    plant_production: pd.DataFrame  # per plant, Wh

    def balance_error_wh(self) -> int:
        """Largest per-interval |alloc + export - production| in Wh. Exactly 0 by construction and by test."""
        diff = self.alloc.sum(axis=1) + self.export - self.production
        return int(diff.abs().max())


def run_allocation(
    community: Community, loads: pd.DataFrame, plants: pd.DataFrame, method: str, shares: np.ndarray | None = None
) -> AllocationResult:
    ids = [m.id for m in community.consumers]
    if list(loads.columns) != ids:
        loads = loads[ids]
    if not loads.index.equals(plants.index):
        raise ValueError("load and PV profiles must share the same 15-minute index")
    if shares is None:
        shares = static_shares(np.array([m.annual_kwh if m.annual_kwh is not None else loads[m.id].sum() / 1000 for m in community.consumers]))
    production = plants.sum(axis=1).astype("int64")
    a = allocate(method, loads.to_numpy(dtype=np.int64), production.to_numpy(dtype=np.int64), shares)
    alloc = pd.DataFrame(a, index=loads.index, columns=ids)
    export = production - alloc.sum(axis=1)
    return AllocationResult(method, loads, alloc, production, export, plants)


def internal_price(params: LegParams) -> float:
    if params.internal_price_chf_kwh is not None:
        return params.internal_price_chf_kwh
    energy = utility_tariff(params.operator_id, "H4")["energy"]
    return params.feed_in_chf_kwh + params.internal_spread_share * max(0.0, energy - params.feed_in_chf_kwh)


def compute_bills(community: Community, res: AllocationResult, params: LegParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (consumer bills, producer revenue), one row per member and month, amounts in CHF."""
    d = params.grid_discount(community)
    share = 1.0 if params.scheme == "zev" else params.discountable_share
    levies = params.levies_on_internal and params.scheme != "zev"
    p_int = internal_price(params)
    month = res.loads.index.to_period("M")

    tot = res.loads.groupby(month).sum() / 1000.0
    internal = res.alloc.groupby(month).sum() / 1000.0
    rows = []
    for m in community.consumers:
        t = utility_tariff(params.operator_id, m.category)
        fixed = t["fixed_chf_month"]
        full = t["grid"] + t["energy"] + t["charge"] + t["aid"]
        int_rate = p_int + t["grid"] * (1 - d * share) + ((t["charge"] + t["aid"]) if levies else 0.0)
        for per in tot.index:
            k_tot, k_int = float(tot.loc[per, m.id]), float(internal.loc[per, m.id])
            k_grid = k_tot - k_int
            bill_leg = k_grid * full + k_int * int_rate + params.admin_fee_chf_month + fixed
            bill_no = k_tot * full + fixed
            rows.append(
                dict(
                    month=str(per), member=m.id, kind=m.kind, kwh_total=k_tot, kwh_community=k_int, kwh_grid=k_grid,
                    rate_grid_chf_kwh=full, rate_community_chf_kwh=int_rate, admin_fee_chf=params.admin_fee_chf_month, fixed_chf=fixed,
                    bill_leg_chf=bill_leg, bill_no_leg_chf=bill_no, saving_chf=bill_no - bill_leg,
                )
            )  # fmt: skip
    consumers = pd.DataFrame(rows)

    # producer revenue: community sales and export are shared between plants in proportion to production per interval
    prod_total = res.production.replace(0, np.nan)
    sold = res.alloc.sum(axis=1)
    prow = []
    for owner in community.producers:
        mine = [p.id for p in community.plants if p.owner == owner.id]
        mine_wh = res.plant_production[mine].sum(axis=1)
        frac = (mine_wh / prod_total).fillna(0.0)
        e_prod = mine_wh.groupby(month).sum() / 1000.0
        e_sold = (sold * frac).groupby(month).sum() / 1000.0
        e_exp = (res.export * frac).groupby(month).sum() / 1000.0
        for per in e_prod.index:
            rev = e_sold[per] * p_int + e_exp[per] * params.feed_in_chf_kwh
            base = e_prod[per] * params.feed_in_chf_kwh
            prow.append(
                dict(
                    month=str(per), member=owner.id, kwh_produced=float(e_prod[per]), kwh_sold_community=float(e_sold[per]),
                    kwh_exported=float(e_exp[per]), revenue_leg_chf=rev, revenue_no_leg_chf=base, gain_chf=rev - base,
                )
            )  # fmt: skip
    return consumers, pd.DataFrame(prow)


def jain_index(x: np.ndarray) -> float:
    """Jain's fairness index: 1 = everyone equal, 1/n = one member gets everything."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or (x**2).sum() == 0:
        return 1.0
    return float(x.sum() ** 2 / (len(x) * (x**2).sum()))


def community_metrics(res: AllocationResult, bills: pd.DataFrame, producers: pd.DataFrame) -> dict:
    prod_kwh = float(res.production.sum()) / 1000
    internal_kwh = float(res.alloc.to_numpy().sum()) / 1000
    cons_kwh = float(res.loads.to_numpy().sum()) / 1000
    per_member = bills.groupby("member").agg(
        saving=("saving_chf", "sum"), no_leg=("bill_no_leg_chf", "sum"), kwh=("kwh_total", "sum"), com=("kwh_community", "sum")
    )
    pct = per_member["saving"] / per_member["no_leg"] * 100
    ss_member = (per_member["com"] / per_member["kwh"]).to_numpy()
    return {
        "method": res.method,
        "production_kwh": prod_kwh,
        "consumption_kwh": cons_kwh,
        "community_kwh": internal_kwh,
        "export_kwh": float(res.export.sum()) / 1000,
        "self_consumption_pct": 100 * internal_kwh / prod_kwh if prod_kwh else 0.0,
        "self_sufficiency_pct": 100 * internal_kwh / cons_kwh if cons_kwh else 0.0,
        "saving_chf_median": float(per_member["saving"].median()),
        "saving_chf_p10": float(per_member["saving"].quantile(0.10)),
        "saving_chf_p90": float(per_member["saving"].quantile(0.90)),
        "saving_chf_min": float(per_member["saving"].min()),
        "saving_chf_max": float(per_member["saving"].max()),
        "saving_chf_total": float(per_member["saving"].sum()),
        "saving_pct_median": float(pct.median()),
        "members_with_saving_pct": 100 * float((per_member["saving"] > 0).mean()),
        "jain_self_sufficiency": jain_index(ss_member),
        "producer_gain_chf_total": float(producers["gain_chf"].sum()) if len(producers) else 0.0,
        "balance_error_wh": res.balance_error_wh(),
    }
