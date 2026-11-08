"""Outputs: CSV, markdown invoices and a meter-data exchange message.

The exchange message is modelled on the structure of the Swiss SDAT-CH data exchange for metered data, which
is the national implementation of the ebIX "validated metered data" report (header, metering point, interval
with resolution, ordered observations with volume and quality). Element names are simplified and the output is
NOT validated against the official schema; see docs/METER-DATA-EXCHANGE.md.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from leg_sim.billing import AllocationResult

NS = "urn:leg-sim:meter-data-exchange:1"


def invoice_markdown(row: pd.Series, community_name: str, discount: float, method: str) -> str:
    """One member, one month, as a markdown invoice."""
    r = row
    lines = [
        f"# Energy statement {r['month']} - {r['member']}",
        "",
        f"Community: {community_name} | allocation method: {method} | grid-fee reduction on community energy: {discount:.0%}",
        "SIMULATED DATA - not a real invoice.",
        "",
        "| Item | Energy (kWh) | Rate (CHF/kWh) | Amount (CHF) |",
        "|---|---:|---:|---:|",
        f"| Community energy (incl. reduced grid fee) | {r['kwh_community']:.3f} | {r['rate_community_chf_kwh']:.4f} | "
        f"{r['kwh_community'] * r['rate_community_chf_kwh']:.2f} |",
        f"| Grid energy (full tariff) | {r['kwh_grid']:.3f} | {r['rate_grid_chf_kwh']:.4f} | "
        f"{r['kwh_grid'] * r['rate_grid_chf_kwh']:.2f} |",
        f"| Fixed charges (metering, basic fee; same with or without community) | | | {r['fixed_chf']:.2f} |",
        f"| Community billing fee | | | {r['admin_fee_chf']:.2f} |",
        f"| **Total with community** | {r['kwh_total']:.3f} | | **{r['bill_leg_chf']:.2f}** |",
        f"| Total without community (all from grid) | {r['kwh_total']:.3f} | {r['rate_grid_chf_kwh']:.4f} | "
        f"{r['bill_no_leg_chf']:.2f} |",
        f"| **Saving** | | | **{r['saving_chf']:.2f}** |",
        "",
    ]
    return "\n".join(lines)


def write_invoices(bills: pd.DataFrame, outdir: Path, community_name: str, discount: float, method: str) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    for _, r in bills.iterrows():
        (outdir / f"{r['member']}_{r['month']}.md").write_text(
            invoice_markdown(r, community_name, discount, method), encoding="utf-8"
        )
    return len(bills)


def meter_data_xml(
    member: str,
    start: pd.Timestamp,
    volumes_wh: pd.Series,
    sender: str = "leg-sim",
    recipient: str = "grid-operator",
    quality: str = "calculated",
    created: str = "2026-01-01T00:00:00Z",
) -> bytes:
    """Build one exchange message: community-allocated energy of a metering point over a contiguous 15-min range."""
    if len(volumes_wh) == 0:
        raise ValueError("no observations to export")
    if (volumes_wh < 0).any():
        raise ValueError("negative volume")
    end = start + pd.Timedelta(minutes=15) * len(volumes_wh)
    root = ET.Element("MeteredDataReport", {"xmlns": NS, "version": "1"})
    hdr = ET.SubElement(root, "HeaderInformation")
    doc = ET.SubElement(hdr, "InstanceDocument")
    ET.SubElement(doc, "DocumentID").text = f"{member}-{start:%Y%m%d%H%M}-{end:%Y%m%d%H%M}"
    ET.SubElement(doc, "Creation").text = created  # fixed default keeps output reproducible
    ET.SubElement(hdr, "ProcessType").text = "allocation-report"
    ET.SubElement(hdr, "SenderIdentification").text = sender
    ET.SubElement(hdr, "RecipientIdentification").text = recipient
    md = ET.SubElement(root, "MeteringData")
    ET.SubElement(md, "MeteringPointID").text = member
    ET.SubElement(md, "ProductType").text = "active-energy-community-allocated"
    iv = ET.SubElement(md, "Interval")
    ET.SubElement(iv, "StartDateTime").text = start.strftime("%Y-%m-%dT%H:%M:%S")
    ET.SubElement(iv, "EndDateTime").text = end.strftime("%Y-%m-%dT%H:%M:%S")
    res = ET.SubElement(md, "Resolution")
    ET.SubElement(res, "Resolution").text = "PT15M"
    ET.SubElement(res, "Unit").text = "KWH"
    for i, wh in enumerate(volumes_wh.to_numpy(), start=1):
        obs = ET.SubElement(md, "Observation")
        ET.SubElement(obs, "Sequence").text = str(i)
        ET.SubElement(obs, "Volume").text = f"{wh / 1000:.3f}"
        ET.SubElement(obs, "Condition").text = quality
    ET.indent(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def write_meter_data(res: AllocationResult, outdir: Path, month: str) -> list[Path]:
    """Write one message per consumer for a month such as '2023-07'."""
    outdir.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
        raise ValueError(f"month must look like YYYY-MM, got {month!r}")
    per = pd.Period(month, freq="M")
    sel = res.alloc[res.alloc.index.to_period("M") == per]
    if sel.empty:
        raise ValueError(f"month {month} is outside the simulated period")
    paths = []
    for member in sel.columns:
        p = outdir / f"{member}_{month}.xml"
        p.write_bytes(meter_data_xml(member, sel.index[0], sel[member]))
        paths.append(p)
    return paths


def parse_meter_data(data: bytes) -> dict:
    """Read a message back (used by tests and as a reference reader)."""
    root = ET.fromstring(data)
    ns = {"n": NS}
    vols = [float(v.text) for v in root.findall(".//n:Observation/n:Volume", ns)]
    seq = [int(v.text) for v in root.findall(".//n:Observation/n:Sequence", ns)]
    return {
        "metering_point": root.findtext(".//n:MeteringPointID", namespaces=ns),
        "start": root.findtext(".//n:Interval/n:StartDateTime", namespaces=ns),
        "end": root.findtext(".//n:Interval/n:EndDateTime", namespaces=ns),
        "resolution": root.findtext(".//n:Resolution/n:Resolution", namespaces=ns),
        "unit": root.findtext(".//n:Resolution/n:Unit", namespaces=ns),
        "volumes_kwh": vols,
        "sequence": seq,
    }
