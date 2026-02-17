"""
Authentication routes and logic for Fantasy Basketball Optimizer.

This module provides user authentication functionality including:
- User registration with email validation
- Login/logout with session management
- Password hashing with bcrypt
- Password change functionality
- Account management

All endpoints return JSON responses and use Flask-Login for session management.
"""

import re
import logging
from datetime import datetime
from functools import wraps

from flask import Blueprint, request, jsonify, current_app
from flask_login import login_user, logout_user, login_required, current_user

from backend.extensions import db, bcrypt
from backend.models import User

# Set up logging
logger = logging.getLogger(__name__)

# Create blueprint
auth_bp = Blueprint('auth', __name__)

# Email validation regex pattern
EMAIL_PATTERN = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)

# Password requirements
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128


# =============================================================================
# Helper Functions
# =============================================================================

def validate_email(email: str) -> tuple[bool, str]:
    """
    Validate email address format.

    Args:
        email: Email address to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not email:
        return False, 'Email is required'

    email = email.strip().lower()

    if len(email) > 255:
        return False, 'Email must be less than 255 characters'

    if not EMAIL_PATTERN.match(email):
        return False, 'Invalid email format'

    return True, ''


def validate_password(password: str) -> tuple[bool, str]:
    """
    Validate password meets requirements.

    Requirements:
    - Minimum 8 characters
    - Maximum 128 characters
    - At least one letter
    - At least one number

    Args:
        password: Password to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not password:
        return False, 'Password is required'

    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f'Password must be at least {MIN_PASSWORD_LENGTH} characters'

    if len(password) > MAX_PASSWORD_LENGTH:
        return False, f'Password must be less than {MAX_PASSWORD_LENGTH} characters'

    if not re.search(r'[a-zA-Z]', password):
        return False, 'Password must contain at least one letter'

    if not re.search(r'\d', password):
        return False, 'Password must contain at least one number'

    return True, ''


def get_client_ip() -> str:
    """Get client IP address from request."""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    return request.remote_addr or 'unknown'


# =============================================================================
# Authentication Endpoints
# =============================================================================

@auth_bp.route('/register', methods=['POST'])
def register():
    """
    Register a new user account.

    Request JSON:
        email (str): User's email address
        password (str): User's password (min 8 chars, must contain letter and number)

    Returns:
        201: User created successfully
            {
                "message": "User registered successfully",
                "user_id": <int>,
                "user": { user object }
            }
        400: Validation error
            { "error": "<error message>" }
        409: Email already registered
            { "error": "Email already registered" }
        500: Server error
            { "error": "Registration failed" }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Extract and validate email
        email = data.get('email', '').strip().lower()
        is_valid, error = validate_email(email)
        if not is_valid:
            return jsonify({'error': error}), 400

        # Extract and validate password
        password = data.get('password', '')
        is_valid, error = validate_password(password)
        if not is_valid:
            return jsonify({'error': error}), 400

        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            logger.warning(f'Registration attempt with existing email: {email}')
            return jsonify({'error': 'Email already registered'}), 409

        # Create new user
        user = User(email=email)
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        logger.info(f'New user registered: {email} from IP {get_client_ip()}')

        return jsonify({
            'message': 'User registered successfully',
            'user_id': user.id,
            'user': user.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        logger.error(f'Registration error: {str(e)}')
        return jsonify({'error': 'Registration failed. Please try again.'}), 500


@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Authenticate a user and create a session.

    Request JSON:
        email (str): User's email address
        password (str): User's password
        remember (bool, optional): Whether to persist session (default: False)

    Returns:
        200: Login successful
            {
                "message": "Login successful",
                "user": { user object }
            }
        400: Missing credentials
            { "error": "Email and password are required" }
        401: Invalid credentials
            { "error": "Invalid email or password" }
        500: Server error
            { "error": "Login failed" }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        remember = data.get('remember', False)

        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400

        # Find user by email
        user = User.query.filter_by(email=email).first()

        # Check password (always check even if user not found to prevent timing attacks)
        if not user:
            # Dummy password check to prevent timing attacks
            bcrypt.check_password_hash(
                '$2b$12$dummy.hash.for.timing.attack.prevention',
                password
            )
            logger.warning(f'Failed login attempt for non-existent user: {email} from IP {get_client_ip()}')
            return jsonify({'error': 'Invalid email or password'}), 401

        if not user.check_password(password):
            logger.warning(f'Failed login attempt for user: {email} from IP {get_client_ip()}')
            return jsonify({'error': 'Invalid email or password'}), 401

        # Update last login timestamp
        user.last_login = datetime.utcnow()
        db.session.commit()

        # Create session
        login_user(user, remember=remember)

        logger.info(f'User logged in: {email} from IP {get_client_ip()}')

        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict()
        }), 200

    except Exception as e:
        logger.error(f'Login error: {str(e)}')
        return jsonify({'error': 'Login failed. Please try again.'}), 500


@auth_bp.route('/logout', methods=['POST'])
@login_required
def logout():
    """
    Log out the current user and end the session.

    Returns:
        200: Logout successful
            { "message": "Logged out successfully" }
    """
    email = current_user.email
    logout_user()
    logger.info(f'User logged out: {email}')

    return jsonify({'message': 'Logged out successfully'}), 200


@auth_bp.route('/verify', methods=['GET'])
def verify():
    """
    Verify if the current session is valid.

    This endpoint can be called without authentication to check session status.

    Returns:
        200: Session status
            {
                "authenticated": <bool>,
                "user": { user object } or null
            }
    """
    if current_user.is_authenticated:
        return jsonify({
            'authenticated': True,
            'user': current_user.to_dict()
        }), 200
    else:
        return jsonify({
            'authenticated': False,
            'user': None
        }), 200


@auth_bp.route('/me', methods=['GET'])
@login_required
def get_current_user():
    """
    Get the current authenticated user's information.

    Returns:
        200: User info
            { "user": { user object } }
        401: Not authenticated
            { "error": "Authentication required" }
    """
    return jsonify({
        'user': current_user.to_dict()
    }), 200


# =============================================================================
# Password Management
# =============================================================================

@auth_bp.route('/change-password', methods=['POST'])
@login_required
def change_password():
    """
    Change the current user's password.

    Request JSON:
        current_password (str): User's current password
        new_password (str): New password (min 8 chars, letter + number)

    Returns:
        200: Password changed successfully
            { "message": "Password changed successfully" }
        400: Validation error
            { "error": "<error message>" }
        401: Current password incorrect
            { "error": "Current password is incorrect" }
        500: Server error
            { "error": "Password change failed" }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        current_password = data.get('current_password', '')
        new_password = data.get('new_password', '')

        if not current_password:
            return jsonify({'error': 'Current password is required'}), 400

        # Validate new password
        is_valid, error = validate_password(new_password)
        if not is_valid:
            return jsonify({'error': error}), 400

        # Verify current password
        if not current_user.check_password(current_password):
            logger.warning(f'Failed password change attempt for user: {current_user.email}')
            return jsonify({'error': 'Current password is incorrect'}), 401

        # Check new password is different
        if current_user.check_password(new_password):
            return jsonify({'error': 'New password must be different from current password'}), 400

        # Update password
        current_user.set_password(new_password)
        db.session.commit()

        logger.info(f'Password changed for user: {current_user.email}')

        return jsonify({'message': 'Password changed successfully'}), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f'Password change error: {str(e)}')
        return jsonify({'error': 'Password change failed. Please try again.'}), 500


# =============================================================================
# Account Management
# =============================================================================

@auth_bp.route('/account', methods=['GET'])
@login_required
def get_account():
    """
    Get the current user's account details.

    Returns:
        200: Account info
            {
                "user": { user object },
                "stats": {
                    "leagues_count": <int>,
                    "member_since": <datetime>
                }
            }
    """
    leagues_count = current_user.leagues.count()

    return jsonify({
        'user': current_user.to_dict(),
        'stats': {
            'leagues_count': leagues_count,
            'member_since': current_user.created_at.isoformat() if current_user.created_at else None
        }
    }), 200


@auth_bp.route('/account', methods=['PUT'])
@login_required
def update_account():
    """
    Update the current user's account details.

    Currently supports updating email only.

    Request JSON:
        email (str, optional): New email address

    Returns:
        200: Account updated
            {
                "message": "Account updated successfully",
                "user": { user object }
            }
        400: Validation error
            { "error": "<error message>" }
        409: Email already in use
            { "error": "Email already in use" }
        500: Server error
            { "error": "Account update failed" }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Update email if provided
        if 'email' in data:
            new_email = data.get('email', '').strip().lower()

            # Validate email
            is_valid, error = validate_email(new_email)
            if not is_valid:
                return jsonify({'error': error}), 400

            # Check if email is different
            if new_email != current_user.email:
                # Check if email is already in use
                existing_user = User.query.filter_by(email=new_email).first()
                if existing_user:
                    return jsonify({'error': 'Email already in use'}), 409

                current_user.email = new_email

        db.session.commit()

        logger.info(f'Account updated for user: {current_user.email}')

        return jsonify({
            'message': 'Account updated successfully',
            'user': current_user.to_dict()
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f'Account update error: {str(e)}')
        return jsonify({'error': 'Account update failed. Please try again.'}), 500


@auth_bp.route('/account', methods=['DELETE'])
@login_required
def delete_account():
    """
    Delete the current user's account.

    This permanently deletes the user and all associated data (leagues, etc.).

    Request JSON:
        password (str): User's password for confirmation

    Returns:
        200: Account deleted
            { "message": "Account deleted successfully" }
        400: Missing password
            { "error": "Password is required to delete account" }
        401: Incorrect password
            { "error": "Incorrect password" }
        500: Server error
            { "error": "Account deletion failed" }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        password = data.get('password', '')

        if not password:
            return jsonify({'error': 'Password is required to delete account'}), 400

        # Verify password
        if not current_user.check_password(password):
            logger.warning(f'Failed account deletion attempt for user: {current_user.email}')
            return jsonify({'error': 'Incorrect password'}), 401

        email = current_user.email
        user_id = current_user.id

        # Log out first
        logout_user()

        # Delete user (cascades to leagues, etc.)
        user = User.query.get(user_id)
        if user:
            db.session.delete(user)
            db.session.commit()

        logger.info(f'Account deleted for user: {email}')

        return jsonify({'message': 'Account deleted successfully'}), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f'Account deletion error: {str(e)}')
        return jsonify({'error': 'Account deletion failed. Please try again.'}), 500


# =============================================================================
# Error Handlers
# =============================================================================

@auth_bp.errorhandler(401)
def unauthorized_handler(error):
    """Handle unauthorized access attempts."""
    return jsonify({
        'error': 'Authentication required',
        'message': 'Please log in to access this resource'
    }), 401


# =============================================================================
# Utility Endpoints
# =============================================================================

@auth_bp.route('/check-email', methods=['POST'])
def check_email():
    """
    Check if an email is already registered.

    Useful for registration form validation.

    Request JSON:
        email (str): Email to check

    Returns:
        200: Email availability
            {
                "available": <bool>,
                "valid": <bool>
            }
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    email = data.get('email', '').strip().lower()

    # Validate format
    is_valid, _ = validate_email(email)

    if not is_valid:
        return jsonify({
            'available': False,
            'valid': False
        }), 200

    # Check if registered
    existing_user = User.query.filter_by(email=email).first()

    return jsonify({
        'available': existing_user is None,
        'valid': True
    }), 200
