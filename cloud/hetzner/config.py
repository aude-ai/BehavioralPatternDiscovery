"""Server configuration using Pydantic settings."""
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # Server
    app_name: str = "BehavioralPatternDiscovery"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # URLs (includes /bpd prefix for nginx-manager routing)
    base_url: str = "https://yourdomain.com/bpd"

    # Database
    database_url: str = "sqlite:///data/db.sqlite"

    # Redis
    redis_url: str = "redis://redis:6379"

    # Storage
    data_dir: str = "/data"
    projects_dir: str = "/data/projects"

    # Modal
    modal_token_id: Optional[str] = None
    modal_token_secret: Optional[str] = None

    # Security
    internal_api_key: str

    # R2 Storage (read-only for status checks)
    r2_access_key_id: Optional[str] = None
    r2_secret_access_key: Optional[str] = None
    r2_endpoint_url: Optional[str] = None
    r2_bucket_name: str = "bpd-storage"

    # LLM APIs
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # CORS
    cors_origins: list[str] = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
