import os
import sys
from pathlib import Path
import pytest

# Ensure project root is importable as a package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Use an isolated SQLite DB for tests to avoid conflicts
os.environ.setdefault("DATABASE_URL", f"sqlite:///{PROJECT_ROOT}/test_py.db")


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    from app.database import init_db
    init_db()
    yield


