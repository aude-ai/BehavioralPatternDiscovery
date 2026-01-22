"""API routes."""
from fastapi import APIRouter

from . import projects, data, pipeline, scoring, patterns, internal, external

api_router = APIRouter()

# Public API routes
api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(data.router, prefix="/projects/{project_id}/data", tags=["data"])
api_router.include_router(pipeline.router, prefix="/projects/{project_id}", tags=["pipeline"])
api_router.include_router(scoring.router, prefix="/projects/{project_id}/scoring", tags=["scoring"])
api_router.include_router(patterns.router, prefix="/projects/{project_id}/patterns", tags=["patterns"])

# Internal routes (for Modal callbacks)
api_router.include_router(internal.router, prefix="/internal", tags=["internal"])

# External API routes
external_router = APIRouter()
external_router.include_router(external.router, prefix="/external", tags=["external"])
