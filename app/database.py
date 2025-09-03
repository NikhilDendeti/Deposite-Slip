from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./test.db')

# For Postgres example: postgresql+psycopg2://user:password@localhost:5432/dbname

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith('sqlite') else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Import Base lazily to avoid circular imports
from .models import Base


def init_db():
    # Create tables
    Base.metadata.create_all(bind=engine)


def migrate_db():
    """Lightweight migration to add new columns if missing (SQLite-safe)."""
    try:
        with engine.begin() as conn:
            inspector = inspect(conn)
            tables = inspector.get_table_names()
            if 'deposit_slips' not in tables:
                return
            existing_cols = {c['name'] for c in inspector.get_columns('deposit_slips')}

            # Add columns if missing
            if 'file_hash' not in existing_cols:
                conn.execute(text("ALTER TABLE deposit_slips ADD COLUMN file_hash VARCHAR"))
            if 'override_reason' not in existing_cols:
                conn.execute(text("ALTER TABLE deposit_slips ADD COLUMN override_reason TEXT"))
            if 'override_approved_by' not in existing_cols:
                conn.execute(text("ALTER TABLE deposit_slips ADD COLUMN override_approved_by VARCHAR"))
            if 'override_at' not in existing_cols:
                conn.execute(text("ALTER TABLE deposit_slips ADD COLUMN override_at DATETIME"))
            if 'processing_details' not in existing_cols:
                conn.execute(text("ALTER TABLE deposit_slips ADD COLUMN processing_details TEXT"))

            # Best-effort index for file_hash (SQLite IF NOT EXISTS supported)
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_deposit_slips_file_hash ON deposit_slips(file_hash)"))
    except Exception:
        # No-op on migration failure to avoid startup crash; errors will surface during queries
        pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



