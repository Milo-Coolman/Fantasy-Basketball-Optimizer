"""
Database initialization script for Fantasy Basketball Optimizer.

This script creates all database tables defined in models.py.
Run from project root: python -m backend.init_db
"""

import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import create_app
from backend.extensions import db
from backend.models import (
    User, League, Team, Player, Roster,
    PlayerStats, Projection, TradeHistory, WaiverRecommendation
)


def init_database():
    """Initialize the database by creating all tables."""
    app = create_app()

    with app.app_context():
        # Create all tables
        db.create_all()

        # Get list of tables created
        inspector = db.inspect(db.engine)
        tables = inspector.get_table_names()

        print("=" * 50)
        print("Database initialized successfully!")
        print("=" * 50)
        print(f"\nDatabase location: {app.config['SQLALCHEMY_DATABASE_URI']}")
        print(f"\nTables created ({len(tables)}):")
        for table in sorted(tables):
            print(f"  - {table}")
        print()


if __name__ == '__main__':
    init_database()
