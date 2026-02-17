"""
Tests for authentication endpoints.

Run with: python -m pytest backend/tests/test_auth.py -v
"""

import pytest
from flask import Flask
from app import create_app
from extensions import db
from models import User
from config import TestingConfig


@pytest.fixture
def app():
    """Create application for testing."""
    app = create_app(TestingConfig)
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False

    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def auth_client(app, client):
    """Create authenticated test client."""
    # Register and login a test user
    client.post('/api/auth/register', json={
        'email': 'test@example.com',
        'password': 'testpass123'
    })
    client.post('/api/auth/login', json={
        'email': 'test@example.com',
        'password': 'testpass123'
    })
    return client


class TestRegistration:
    """Test user registration endpoint."""

    def test_register_success(self, client):
        """Test successful user registration."""
        response = client.post('/api/auth/register', json={
            'email': 'newuser@example.com',
            'password': 'password123'
        })

        assert response.status_code == 201
        data = response.get_json()
        assert data['message'] == 'User registered successfully'
        assert 'user_id' in data
        assert data['user']['email'] == 'newuser@example.com'

    def test_register_missing_email(self, client):
        """Test registration with missing email."""
        response = client.post('/api/auth/register', json={
            'password': 'password123'
        })

        assert response.status_code == 400
        assert 'error' in response.get_json()

    def test_register_missing_password(self, client):
        """Test registration with missing password."""
        response = client.post('/api/auth/register', json={
            'email': 'test@example.com'
        })

        assert response.status_code == 400
        assert 'error' in response.get_json()

    def test_register_invalid_email(self, client):
        """Test registration with invalid email format."""
        response = client.post('/api/auth/register', json={
            'email': 'notanemail',
            'password': 'password123'
        })

        assert response.status_code == 400
        assert 'Invalid email' in response.get_json()['error']

    def test_register_short_password(self, client):
        """Test registration with too short password."""
        response = client.post('/api/auth/register', json={
            'email': 'test@example.com',
            'password': 'short'
        })

        assert response.status_code == 400
        assert 'at least 8 characters' in response.get_json()['error']

    def test_register_password_no_letter(self, client):
        """Test registration with password missing letters."""
        response = client.post('/api/auth/register', json={
            'email': 'test@example.com',
            'password': '12345678'
        })

        assert response.status_code == 400
        assert 'letter' in response.get_json()['error']

    def test_register_password_no_number(self, client):
        """Test registration with password missing numbers."""
        response = client.post('/api/auth/register', json={
            'email': 'test@example.com',
            'password': 'abcdefgh'
        })

        assert response.status_code == 400
        assert 'number' in response.get_json()['error']

    def test_register_duplicate_email(self, client):
        """Test registration with already registered email."""
        # Register first user
        client.post('/api/auth/register', json={
            'email': 'test@example.com',
            'password': 'password123'
        })

        # Try to register again with same email
        response = client.post('/api/auth/register', json={
            'email': 'test@example.com',
            'password': 'password456'
        })

        assert response.status_code == 409
        assert 'already registered' in response.get_json()['error']

    def test_register_email_case_insensitive(self, client):
        """Test that email is case insensitive."""
        client.post('/api/auth/register', json={
            'email': 'Test@Example.com',
            'password': 'password123'
        })

        response = client.post('/api/auth/register', json={
            'email': 'test@example.com',
            'password': 'password456'
        })

        assert response.status_code == 409


class TestLogin:
    """Test user login endpoint."""

    def test_login_success(self, client):
        """Test successful login."""
        # Register user first
        client.post('/api/auth/register', json={
            'email': 'test@example.com',
            'password': 'password123'
        })

        response = client.post('/api/auth/login', json={
            'email': 'test@example.com',
            'password': 'password123'
        })

        assert response.status_code == 200
        data = response.get_json()
        assert data['message'] == 'Login successful'
        assert data['user']['email'] == 'test@example.com'

    def test_login_wrong_password(self, client):
        """Test login with wrong password."""
        client.post('/api/auth/register', json={
            'email': 'test@example.com',
            'password': 'password123'
        })

        response = client.post('/api/auth/login', json={
            'email': 'test@example.com',
            'password': 'wrongpassword'
        })

        assert response.status_code == 401
        assert 'Invalid email or password' in response.get_json()['error']

    def test_login_nonexistent_user(self, client):
        """Test login with non-existent user."""
        response = client.post('/api/auth/login', json={
            'email': 'nonexistent@example.com',
            'password': 'password123'
        })

        assert response.status_code == 401
        assert 'Invalid email or password' in response.get_json()['error']

    def test_login_missing_credentials(self, client):
        """Test login with missing credentials."""
        response = client.post('/api/auth/login', json={})

        assert response.status_code == 400

    def test_login_email_case_insensitive(self, client):
        """Test that login email is case insensitive."""
        client.post('/api/auth/register', json={
            'email': 'test@example.com',
            'password': 'password123'
        })

        response = client.post('/api/auth/login', json={
            'email': 'TEST@EXAMPLE.COM',
            'password': 'password123'
        })

        assert response.status_code == 200


class TestLogout:
    """Test user logout endpoint."""

    def test_logout_success(self, auth_client):
        """Test successful logout."""
        response = auth_client.post('/api/auth/logout')

        assert response.status_code == 200
        assert 'Logged out' in response.get_json()['message']

    def test_logout_not_authenticated(self, client):
        """Test logout when not authenticated."""
        response = client.post('/api/auth/logout')

        # Should redirect or return 401
        assert response.status_code in [401, 302]


class TestVerify:
    """Test session verification endpoint."""

    def test_verify_authenticated(self, auth_client):
        """Test verify when authenticated."""
        response = auth_client.get('/api/auth/verify')

        assert response.status_code == 200
        data = response.get_json()
        assert data['authenticated'] is True
        assert data['user'] is not None

    def test_verify_not_authenticated(self, client):
        """Test verify when not authenticated."""
        response = client.get('/api/auth/verify')

        assert response.status_code == 200
        data = response.get_json()
        assert data['authenticated'] is False
        assert data['user'] is None


class TestCurrentUser:
    """Test get current user endpoint."""

    def test_get_current_user_authenticated(self, auth_client):
        """Test getting current user when authenticated."""
        response = auth_client.get('/api/auth/me')

        assert response.status_code == 200
        data = response.get_json()
        assert data['user']['email'] == 'test@example.com'

    def test_get_current_user_not_authenticated(self, client):
        """Test getting current user when not authenticated."""
        response = client.get('/api/auth/me')

        assert response.status_code in [401, 302]


class TestChangePassword:
    """Test password change endpoint."""

    def test_change_password_success(self, auth_client):
        """Test successful password change."""
        response = auth_client.post('/api/auth/change-password', json={
            'current_password': 'testpass123',
            'new_password': 'newpass456'
        })

        assert response.status_code == 200
        assert 'Password changed' in response.get_json()['message']

        # Verify new password works
        auth_client.post('/api/auth/logout')
        login_response = auth_client.post('/api/auth/login', json={
            'email': 'test@example.com',
            'password': 'newpass456'
        })
        assert login_response.status_code == 200

    def test_change_password_wrong_current(self, auth_client):
        """Test password change with wrong current password."""
        response = auth_client.post('/api/auth/change-password', json={
            'current_password': 'wrongpassword',
            'new_password': 'newpass456'
        })

        assert response.status_code == 401

    def test_change_password_same_password(self, auth_client):
        """Test password change with same password."""
        response = auth_client.post('/api/auth/change-password', json={
            'current_password': 'testpass123',
            'new_password': 'testpass123'
        })

        assert response.status_code == 400
        assert 'different' in response.get_json()['error']

    def test_change_password_invalid_new_password(self, auth_client):
        """Test password change with invalid new password."""
        response = auth_client.post('/api/auth/change-password', json={
            'current_password': 'testpass123',
            'new_password': 'short'
        })

        assert response.status_code == 400


class TestAccount:
    """Test account management endpoints."""

    def test_get_account(self, auth_client):
        """Test getting account details."""
        response = auth_client.get('/api/auth/account')

        assert response.status_code == 200
        data = response.get_json()
        assert 'user' in data
        assert 'stats' in data
        assert 'leagues_count' in data['stats']

    def test_update_account_email(self, auth_client):
        """Test updating account email."""
        response = auth_client.put('/api/auth/account', json={
            'email': 'newemail@example.com'
        })

        assert response.status_code == 200
        data = response.get_json()
        assert data['user']['email'] == 'newemail@example.com'

    def test_update_account_duplicate_email(self, client):
        """Test updating account with duplicate email."""
        # Create two users
        client.post('/api/auth/register', json={
            'email': 'user1@example.com',
            'password': 'password123'
        })
        client.post('/api/auth/register', json={
            'email': 'user2@example.com',
            'password': 'password123'
        })

        # Login as user1
        client.post('/api/auth/login', json={
            'email': 'user1@example.com',
            'password': 'password123'
        })

        # Try to change to user2's email
        response = client.put('/api/auth/account', json={
            'email': 'user2@example.com'
        })

        assert response.status_code == 409

    def test_delete_account_success(self, auth_client, app):
        """Test successful account deletion."""
        response = auth_client.delete('/api/auth/account', json={
            'password': 'testpass123'
        })

        assert response.status_code == 200

        # Verify user is deleted
        with app.app_context():
            user = User.query.filter_by(email='test@example.com').first()
            assert user is None

    def test_delete_account_wrong_password(self, auth_client):
        """Test account deletion with wrong password."""
        response = auth_client.delete('/api/auth/account', json={
            'password': 'wrongpassword'
        })

        assert response.status_code == 401


class TestCheckEmail:
    """Test email availability check endpoint."""

    def test_check_email_available(self, client):
        """Test checking available email."""
        response = client.post('/api/auth/check-email', json={
            'email': 'available@example.com'
        })

        assert response.status_code == 200
        data = response.get_json()
        assert data['available'] is True
        assert data['valid'] is True

    def test_check_email_taken(self, client):
        """Test checking taken email."""
        client.post('/api/auth/register', json={
            'email': 'taken@example.com',
            'password': 'password123'
        })

        response = client.post('/api/auth/check-email', json={
            'email': 'taken@example.com'
        })

        assert response.status_code == 200
        data = response.get_json()
        assert data['available'] is False
        assert data['valid'] is True

    def test_check_email_invalid(self, client):
        """Test checking invalid email format."""
        response = client.post('/api/auth/check-email', json={
            'email': 'notanemail'
        })

        assert response.status_code == 200
        data = response.get_json()
        assert data['available'] is False
        assert data['valid'] is False
