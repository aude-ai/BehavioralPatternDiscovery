"""FastAPI application entry point."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .database import init_db
from .routes import api_router, internal_router, external_router
from .websocket import ws_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Initializing database...")
    init_db()
    logger.info("Application started")
    yield
    # Shutdown
    logger.info("Application shutdown")


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(api_router, prefix="/api")
app.include_router(internal_router)  # At /internal (Modal callbacks)
app.include_router(external_router)  # At /external (integration API)


# WebSocket endpoint
@app.websocket("/ws/projects/{project_id}/events")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    """WebSocket for real-time project updates."""
    await ws_manager.connect(project_id, websocket)
    try:
        while True:
            # Keep connection alive, handle pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await ws_manager.disconnect(project_id, websocket)


# Health check
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Frontend config endpoint
@app.get("/api/config")
def get_frontend_config():
    """Get frontend configuration values from scoring.yaml."""
    from pathlib import Path
    from src.core.config import load_config

    config_path = Path(__file__).parent.parent.parent / "config" / "scoring.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Required config file not found: {config_path}")

    scoring_config = load_config(config_path)

    # Validate required config structure
    if "report" not in scoring_config:
        raise ValueError("Missing required config section: report in scoring.yaml")
    if "content" not in scoring_config["report"]:
        raise ValueError("Missing required config section: report.content in scoring.yaml")

    report_content = scoring_config["report"]["content"]

    # Return frontend-relevant config values (with documented defaults)
    return {
        "strength_threshold": report_content.get("strength_threshold", 70),
        "weakness_threshold": report_content.get("weakness_threshold", 40),
    }


# Serve frontend (if exists)
try:
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
except Exception:
    logger.warning("Frontend not found, skipping static file serving")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
