"""Command line interface: ``python -m leg_sim <command>``."""

from __future__ import annotations

from pathlib import Path

import typer

from leg_sim import community as comm
from leg_sim.allocation import METHODS
from leg_sim.billing import community_metrics, compute_bills, run_allocation
from leg_sim.demo import format_summary, run_demo
from leg_sim.export import write_invoices, write_meter_data
from leg_sim.models import Community, LegParams

app = typer.Typer(help="LEG / ZEV allocation, tariff and billing engine (simulated demo data).", no_args_is_help=True)


@app.command()
def demo(
    out: Path = typer.Option(Path("out"), help="output directory"),
    docs: Path = typer.Option(None, help="if set, also write docs/figure.png and docs/post-figure.png here"),
    seed: int = typer.Option(7, help="random seed of the simulated load profiles"),
) -> None:
    """Simulate the demo community for 2023, run all methods, write bills, invoices, XML and print the results."""
    s = run_demo(out, docs, seed)
    typer.echo(format_summary(s))
    typer.echo(f"\nFiles written to {out}/")


@app.command()
def simulate(
    community: Path = typer.Option(..., exists=True, help="community.json (see out/community.json from the demo)"),
    loads: Path = typer.Option(..., exists=True, help="15-minute Wh CSV, one column per consumer"),
    pv: Path = typer.Option(..., exists=True, help="15-minute Wh CSV, one column per PV plant"),
    method: str = typer.Option("fair", help=f"one of {', '.join(METHODS)}"),
    out: Path = typer.Option(Path("out_sim")),
    operator_id: int = typer.Option(486, help="ElCom operator id for the tariff"),
    discount: float = typer.Option(None, help="override the grid-fee reduction (0..1)"),
    scheme: str = typer.Option("leg", help="leg or zev"),
) -> None:
    """Run one allocation method on your own meter data and write bills and invoices."""
    if method not in METHODS:
        raise typer.BadParameter(f"method must be one of {METHODS}")
    try:
        c = Community.model_validate_json(community.read_text(encoding="utf-8"))
        ids = [m.id for m in c.consumers]
        ld = comm.read_profiles(loads, ids)
        pl = comm.read_profiles(pv, [p.id for p in c.plants])
        params = LegParams(operator_id=operator_id, discount_override=discount, scheme=scheme)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    for m in c.consumers:
        if m.annual_kwh is None:
            m.annual_kwh = float(ld[m.id].sum()) / 1000.0
    res = run_allocation(c, ld, pl, method)
    bills, prod = compute_bills(c, res, params)
    out.mkdir(parents=True, exist_ok=True)
    bills.to_csv(out / "bills_monthly.csv", index=False, float_format="%.4f")
    prod.to_csv(out / "producer_revenue_monthly.csv", index=False, float_format="%.4f")
    write_invoices(bills, out / "invoices", c.name, params.grid_discount(c), method)
    m = community_metrics(res, bills, prod)
    typer.echo(
        f"{method}: self-consumption {m['self_consumption_pct']:.1f}%, self-sufficiency {m['self_sufficiency_pct']:.1f}%, "
        f"median member saving CHF {m['saving_chf_median']:.0f}/yr, balance error {m['balance_error_wh']} Wh"
    )


@app.command("export-meter-data")
def export_meter_data(
    community: Path = typer.Option(..., exists=True),
    loads: Path = typer.Option(..., exists=True),
    pv: Path = typer.Option(..., exists=True),
    month: str = typer.Option(..., help="YYYY-MM"),
    method: str = typer.Option("fair"),
    out: Path = typer.Option(Path("out_xml")),
) -> None:
    """Write one SDAT-CH/ebIX-style metered-data message per consumer for a month (see docs/METER-DATA-EXCHANGE.md)."""
    try:
        c = Community.model_validate_json(community.read_text(encoding="utf-8"))
        ld = comm.read_profiles(loads, [m.id for m in c.consumers])
        pl = comm.read_profiles(pv, [p.id for p in c.plants])
        for m in c.consumers:
            m.annual_kwh = m.annual_kwh or float(ld[m.id].sum()) / 1000.0
        res = run_allocation(c, ld, pl, method)
        paths = write_meter_data(res, out, month)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"wrote {len(paths)} messages to {out}/")


@app.command("fetch-data")
def fetch_data() -> None:
    """Re-download the committed open-data extracts (PVGIS PV yield, ElCom 2026 tariffs). Needs internet."""
    from leg_sim import opendata

    typer.echo(opendata.fetch_pvgis())
    typer.echo(opendata.fetch_elcom())


if __name__ == "__main__":
    app()
