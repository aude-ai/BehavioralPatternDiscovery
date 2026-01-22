"""WebSocket manager for real-time updates."""
import asyncio
import json
import logging
from typing import Dict, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections per project."""

    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, project_id: str, websocket: WebSocket):
        """Connect a WebSocket to a project."""
        await websocket.accept()
        async with self._lock:
            if project_id not in self.connections:
                self.connections[project_id] = set()
            self.connections[project_id].add(websocket)
        logger.info(f"WebSocket connected for project {project_id}")

    async def disconnect(self, project_id: str, websocket: WebSocket):
        """Disconnect a WebSocket from a project."""
        async with self._lock:
            if project_id in self.connections:
                self.connections[project_id].discard(websocket)
                if not self.connections[project_id]:
                    del self.connections[project_id]
        logger.info(f"WebSocket disconnected for project {project_id}")

    async def broadcast(self, project_id: str, message: dict):
        """Broadcast message to all connections for a project."""
        if project_id not in self.connections:
            return

        dead_connections = set()
        message_json = json.dumps(message)

        for websocket in self.connections[project_id]:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                dead_connections.add(websocket)

        # Clean up dead connections
        if dead_connections:
            async with self._lock:
                if project_id in self.connections:
                    self.connections[project_id] -= dead_connections


# Global instance
ws_manager = WebSocketManager()


async def broadcast_to_project(project_id: str, message: dict):
    """Convenience function to broadcast to a project."""
    await ws_manager.broadcast(project_id, message)
