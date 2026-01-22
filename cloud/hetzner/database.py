"""Database and Redis connections."""
import json
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, Optional
from uuid import uuid4

import redis
from sqlalchemy import create_engine, Column, String, DateTime, Float, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from .config import get_settings

settings = get_settings()

# SQLAlchemy setup
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis setup
redis_client = redis.from_url(settings.redis_url, decode_responses=True)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Get database session context manager."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_redis() -> redis.Redis:
    """Get Redis client."""
    return redis_client


# Database models
class ProjectModel(Base):
    """Project database model."""

    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    config_overrides = Column(JSON, nullable=True)
    status = Column(String, default="created")
    metadata_ = Column("metadata", JSON, nullable=True)


class JobModel(Base):
    """Job database model."""

    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id = Column(String, nullable=False, index=True)
    job_type = Column(String, nullable=False)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    progress = Column(Float, default=0.0)
    progress_message = Column(String, nullable=True)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    modal_call_id = Column(String, nullable=True)
    metadata_ = Column("metadata", JSON, nullable=True)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
