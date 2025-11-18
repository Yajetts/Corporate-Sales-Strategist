"""Database migration utilities"""

import logging
from alembic import command
from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
from sqlalchemy import inspect
from src.api.database import postgres_manager, Base
import os

logger = logging.getLogger(__name__)


class MigrationManager:
    """Manages database schema migrations"""
    
    def __init__(self):
        """Initialize migration manager"""
        self.alembic_cfg = None
        self._setup_alembic()
    
    def _setup_alembic(self):
        """Setup Alembic configuration"""
        # Create alembic.ini if it doesn't exist
        alembic_ini_path = 'alembic.ini'
        if not os.path.exists(alembic_ini_path):
            logger.warning("alembic.ini not found. Using default configuration.")
            # Use programmatic configuration
            self.alembic_cfg = AlembicConfig()
            self.alembic_cfg.set_main_option('script_location', 'alembic')
        else:
            self.alembic_cfg = AlembicConfig(alembic_ini_path)
    
    def create_migration(self, message: str):
        """
        Create a new migration script.
        
        Args:
            message: Migration description
        """
        try:
            command.revision(self.alembic_cfg, message=message, autogenerate=True)
            logger.info(f"Created migration: {message}")
        except Exception as e:
            logger.error(f"Failed to create migration: {e}")
            raise
    
    def upgrade(self, revision: str = 'head'):
        """
        Upgrade database to a specific revision.
        
        Args:
            revision: Target revision (default: 'head' for latest)
        """
        try:
            command.upgrade(self.alembic_cfg, revision)
            logger.info(f"Upgraded database to revision: {revision}")
        except Exception as e:
            logger.error(f"Failed to upgrade database: {e}")
            raise
    
    def downgrade(self, revision: str):
        """
        Downgrade database to a specific revision.
        
        Args:
            revision: Target revision
        """
        try:
            command.downgrade(self.alembic_cfg, revision)
            logger.info(f"Downgraded database to revision: {revision}")
        except Exception as e:
            logger.error(f"Failed to downgrade database: {e}")
            raise
    
    def current_revision(self) -> str:
        """
        Get current database revision.
        
        Returns:
            Current revision identifier
        """
        try:
            postgres_manager.initialize()
            with postgres_manager.engine.connect() as connection:
                context = MigrationContext.configure(connection)
                current = context.get_current_revision()
                return current if current else 'No revision'
        except Exception as e:
            logger.error(f"Failed to get current revision: {e}")
            return 'Unknown'
    
    def pending_migrations(self) -> list:
        """
        Get list of pending migrations.
        
        Returns:
            List of pending migration revisions
        """
        try:
            script = ScriptDirectory.from_config(self.alembic_cfg)
            current = self.current_revision()
            
            if current == 'No revision':
                # All migrations are pending
                return [rev.revision for rev in script.walk_revisions()]
            
            pending = []
            for rev in script.walk_revisions():
                if rev.revision != current:
                    pending.append(rev.revision)
                else:
                    break
            
            return pending
        except Exception as e:
            logger.error(f"Failed to get pending migrations: {e}")
            return []


def create_tables_if_not_exist():
    """
    Create all tables if they don't exist.
    This is a simple alternative to migrations for development.
    """
    try:
        postgres_manager.initialize()
        
        # Check if tables exist
        inspector = inspect(postgres_manager.engine)
        existing_tables = inspector.get_table_names()
        
        if not existing_tables:
            logger.info("No tables found. Creating all tables...")
            Base.metadata.create_all(postgres_manager.engine)
            logger.info("All tables created successfully")
        else:
            logger.info(f"Found existing tables: {existing_tables}")
            # Create only missing tables
            Base.metadata.create_all(postgres_manager.engine, checkfirst=True)
            logger.info("Verified all tables exist")
        
        return True
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        return False


def drop_all_tables():
    """
    Drop all tables. USE WITH CAUTION!
    This is primarily for testing and development.
    """
    try:
        postgres_manager.initialize()
        logger.warning("Dropping all tables...")
        Base.metadata.drop_all(postgres_manager.engine)
        logger.info("All tables dropped")
        return True
    except Exception as e:
        logger.error(f"Failed to drop tables: {e}")
        return False


# CLI-friendly functions
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python migrations.py [create|upgrade|downgrade|current|pending]")
        sys.exit(1)
    
    manager = MigrationManager()
    command_name = sys.argv[1]
    
    if command_name == 'create':
        if len(sys.argv) < 3:
            print("Usage: python migrations.py create <message>")
            sys.exit(1)
        message = ' '.join(sys.argv[2:])
        manager.create_migration(message)
    
    elif command_name == 'upgrade':
        revision = sys.argv[2] if len(sys.argv) > 2 else 'head'
        manager.upgrade(revision)
    
    elif command_name == 'downgrade':
        if len(sys.argv) < 3:
            print("Usage: python migrations.py downgrade <revision>")
            sys.exit(1)
        manager.downgrade(sys.argv[2])
    
    elif command_name == 'current':
        print(f"Current revision: {manager.current_revision()}")
    
    elif command_name == 'pending':
        pending = manager.pending_migrations()
        if pending:
            print(f"Pending migrations: {', '.join(pending)}")
        else:
            print("No pending migrations")
    
    elif command_name == 'init':
        create_tables_if_not_exist()
    
    elif command_name == 'drop':
        confirm = input("Are you sure you want to drop all tables? (yes/no): ")
        if confirm.lower() == 'yes':
            drop_all_tables()
        else:
            print("Operation cancelled")
    
    else:
        print(f"Unknown command: {command_name}")
        sys.exit(1)
