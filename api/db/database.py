from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os
from dotenv import load_dotenv

load_dotenv(".env")


from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os
from dotenv import load_dotenv

load_dotenv(".env")

Base = declarative_base()

_engine = None


def get_db_engine():
    """Lazy engine: import-safe when DB_URL is missing (tests, lint)."""
    global _engine
    if _engine is not None:
        return _engine

    DATABASE_URL = os.getenv("DB_URL")

    if not DATABASE_URL:
        # Import-safe fallback so `from api.v1.models import ...` works
        # without a live Postgres. Real app requires DB_URL (see create_database).
        DATABASE_URL = "sqlite:///./learnly_fallback.db"

    connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
    pool_kwargs = {} if DATABASE_URL.startswith("sqlite") else {"pool_size": 32, "max_overflow": 64}
    _engine = create_engine(DATABASE_URL, connect_args=connect_args, **pool_kwargs)
    return _engine


# Back-compat: some modules import db_engine directly.
def __getattr__(name: str):
    if name == "db_engine":
        return get_db_engine()
    raise AttributeError(name)


# Session and Base declaration
SessionLocal = sessionmaker(autocommit=False, autoflush=False)


def _bind_session():
    if SessionLocal.kw.get("bind") is None:
        SessionLocal.configure(bind=get_db_engine())


def create_database():
    return Base.metadata.create_all(bind=get_db_engine())


def get_db():
    _bind_session()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
