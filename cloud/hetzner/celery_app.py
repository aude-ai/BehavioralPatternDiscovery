"""Celery application configuration."""
from celery import Celery

from .config import get_settings

settings = get_settings()

celery_app = Celery(
    "bpd",
    broker=f"{settings.redis_url}/0",
    backend=f"{settings.redis_url}/1",
    include=[
        "cloud.hetzner.tasks.cpu_tasks",
        "cloud.hetzner.tasks.gpu_tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=86400,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_routes={
        "cloud.hetzner.tasks.cpu_tasks.*": {"queue": "cpu"},
        "cloud.hetzner.tasks.gpu_tasks.*": {"queue": "gpu"},
    },
)
