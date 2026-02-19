"""
Flask extension instances.

This module creates extension instances without initializing them.
They are initialized in the application factory (app.py).
This avoids circular import issues.
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate

# Database
db = SQLAlchemy()

# Database migrations
migrate = Migrate()

# Authentication
login_manager = LoginManager()

# Password hashing
bcrypt = Bcrypt()
