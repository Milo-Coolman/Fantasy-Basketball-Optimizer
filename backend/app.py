"""
Fantasy Basketball Optimizer - Flask Application

Main entry point for the Flask backend API.
"""

import atexit
import os
import logging
from logging.handlers import RotatingFileHandler

from flask import Flask, jsonify
from flask_cors import CORS

from backend.extensions import db, login_manager, bcrypt, migrate
from backend.config import get_config
from backend.services.scheduler_service import init_scheduler, shutdown_scheduler, get_scheduler


def create_app(config_class=None):
    """
    Application factory for creating Flask app instances.

    Args:
        config_class: Configuration class to use. If None, uses get_config().

    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)

    # Load configuration
    if config_class is None:
        config_class = get_config()
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    bcrypt.init_app(app)

    # Configure CORS
    CORS(
        app,
        origins=app.config.get('CORS_ORIGINS', ['http://localhost:3000']),
        supports_credentials=app.config.get('CORS_SUPPORTS_CREDENTIALS', True),
        allow_headers=['Content-Type', 'Authorization'],
        methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
    )

    # Configure login manager
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'

    # Configure logging
    configure_logging(app)

    # Register blueprints
    register_blueprints(app)

    # Register error handlers
    register_error_handlers(app)

    # Register user loader
    register_user_loader()

    # Create database tables
    with app.app_context():
        db.create_all()

    # Initialize and start scheduler (only if not in testing mode)
    if not app.testing:
        init_scheduler(app)
        # Register shutdown handler
        atexit.register(shutdown_scheduler)

    app.logger.info('Fantasy Basketball Optimizer started successfully')

    return app


def configure_logging(app):
    """Configure application logging."""
    if not app.debug and not app.testing:
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.mkdir('logs')

        # File handler for production
        file_handler = RotatingFileHandler(
            'logs/fantasy_basketball.log',
            maxBytes=10240000,  # 10 MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)

    # Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    console_handler.setLevel(logging.DEBUG if app.debug else logging.INFO)
    app.logger.addHandler(console_handler)

    app.logger.setLevel(logging.DEBUG if app.debug else logging.INFO)


def register_blueprints(app):
    """Register Flask blueprints for API routes."""
    from backend.auth import auth_bp
    from backend.api.leagues import leagues_bp
    from backend.api.projections import projections_bp
    from backend.api.trades import trades_bp
    from backend.api.waivers import waivers_bp
    from backend.api.dashboard import dashboard_bp

    # Register blueprints with URL prefixes
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(leagues_bp, url_prefix='/api/leagues')
    app.register_blueprint(projections_bp, url_prefix='/api')
    app.register_blueprint(trades_bp, url_prefix='/api')
    app.register_blueprint(waivers_bp, url_prefix='/api')
    app.register_blueprint(dashboard_bp, url_prefix='/api')


def register_error_handlers(app):
    """Register error handlers for common HTTP errors."""

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'error': 'Bad Request',
            'message': str(error.description) if hasattr(error, 'description') else 'Invalid request'
        }), 400

    @app.errorhandler(401)
    def unauthorized(error):
        return jsonify({
            'error': 'Unauthorized',
            'message': 'Authentication required'
        }), 401

    @app.errorhandler(403)
    def forbidden(error):
        return jsonify({
            'error': 'Forbidden',
            'message': 'You do not have permission to access this resource'
        }), 403

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Not Found',
            'message': 'The requested resource was not found'
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        app.logger.error(f'Internal server error: {error}')
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred'
        }), 500

    # Health check endpoint
    @app.route('/api/health')
    def health_check():
        """Health check endpoint for monitoring."""
        scheduler = get_scheduler()
        return jsonify({
            'status': 'healthy',
            'version': '1.0.0',
            'scheduler_running': scheduler.is_running if scheduler else False
        })

    # Scheduler status endpoint
    @app.route('/api/scheduler/status')
    def scheduler_status():
        """Get scheduler status and job information."""
        scheduler = get_scheduler()
        if not scheduler or not scheduler.is_running:
            return jsonify({
                'running': False,
                'jobs': []
            })

        return jsonify({
            'running': True,
            'jobs': scheduler.get_jobs()
        })


def register_user_loader():
    """Register user loader callback for Flask-Login."""
    @login_manager.user_loader
    def load_user(user_id):
        from backend.models import User
        return User.query.get(int(user_id))


# Create application instance for Flask CLI and development server
app = create_app()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
