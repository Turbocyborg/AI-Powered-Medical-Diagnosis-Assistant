"""
Initialize database and create tables
"""

from database import init_db, engine
from models import Base
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Initialize database"""
    try:
        logger.info("Creating database tables...")
        init_db()
        logger.info("✓ Database tables created successfully!")

        # List all tables
        from sqlalchemy import inspect

        inspector = inspect(engine)
        tables = inspector.get_table_names()

        logger.info(f"✓ Created {len(tables)} tables:")
        for table in tables:
            logger.info(f"  - {table}")

    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}")
        raise


if __name__ == "__main__":
    main()
