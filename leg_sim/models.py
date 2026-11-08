"""Pydantic models: community structure and billing parameters."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Role = Literal["consumer", "producer"]
Category = Literal["H4", "C2"]  # ElCom tariff categories (4-room household, small business)
Scheme = Literal["leg", "zev"]


class Building(BaseModel):
    id: str
    transformer_station: str = Field(description="Transformer station the building is connected behind")


class Member(BaseModel):
    id: str
    kind: str = Field(description="free text, e.g. household, bakery, common_area, pv_owner")
    role: Role = "consumer"
    building: str
    category: Category = "H4"
    annual_kwh: float | None = Field(default=None, ge=0, description="expected yearly consumption (static shares)")

    @field_validator("id")
    @classmethod
    def _id_safe(cls, v: str) -> str:
        if not v or not all(c.isalnum() or c in "_-" for c in v):
            raise ValueError("member id must be non-empty and contain only letters, digits, '_' or '-'")
        return v


class Plant(BaseModel):
    id: str
    building: str
    owner: str = Field(description="member id of the producer that owns the plant")
    orientation: Literal["south", "east", "west"] = "south"
    kwp: float = Field(gt=0)


class Community(BaseModel):
    name: str = "demo-community"
    buildings: list[Building]
    members: list[Member]
    plants: list[Plant]

    @model_validator(mode="after")
    def _consistent(self) -> Community:
        ids = [m.id for m in self.members]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate member ids")
        bids = {b.id for b in self.buildings}
        for m in self.members:
            if m.building not in bids:
                raise ValueError(f"member {m.id}: unknown building {m.building}")
        owners = {m.id for m in self.members if m.role == "producer"}
        for p in self.plants:
            if p.building not in bids:
                raise ValueError(f"plant {p.id}: unknown building {p.building}")
            if p.owner not in owners:
                raise ValueError(f"plant {p.id}: owner {p.owner} is not a producer member")
        if not self.consumers:
            raise ValueError("community needs at least one consumer")
        return self

    @property
    def consumers(self) -> list[Member]:
        return [m for m in self.members if m.role == "consumer"]

    @property
    def producers(self) -> list[Member]:
        return [m for m in self.members if m.role == "producer"]

    @property
    def n_transformer_stations(self) -> int:
        used = {b.transformer_station for b in self.buildings if any(m.building == b.id for m in self.members)}
        return len(used)


class LegParams(BaseModel):
    """Billing parameters. Every legal number is a parameter, not a hard-coded fact (see README, Legal basis)."""

    scheme: Scheme = "leg"
    operator_id: int = Field(default=486, description="ElCom operator id whose 2026 tariff applies")
    discount_single_ts: float = Field(default=0.40, ge=0, le=1, description="grid-fee reduction, one transformer station")
    discount_multi_ts: float = Field(default=0.20, ge=0, le=1, description="grid-fee reduction, several stations")
    discount_override: float | None = Field(default=None, ge=0, le=1)
    discountable_share: float = Field(
        default=0.9, ge=0, le=1, description="share of ElCom 'grid usage' that the reduction applies to (assumption)"
    )
    levies_on_internal: bool = Field(default=True, description="federal/municipal levies and grid surcharge still due")
    feed_in_chf_kwh: float = Field(default=0.08, ge=0, description="remuneration for exported PV energy (assumption)")
    internal_price_chf_kwh: float | None = Field(default=None, ge=0, description="community energy price; None = split")
    internal_spread_share: float = Field(default=0.5, ge=0, le=1, description="share of (energy tariff - feed-in) charged")
    admin_fee_chf_month: float = Field(default=0.0, ge=0, description="community billing fee per consumer (assumption)")

    def grid_discount(self, community: Community) -> float:
        if self.discount_override is not None:
            return self.discount_override
        if self.scheme == "zev":
            return 1.0
        return self.discount_single_ts if community.n_transformer_stations <= 1 else self.discount_multi_ts
