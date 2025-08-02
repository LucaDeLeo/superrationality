"""
WebSocket connection manager for real-time updates.
"""
import json
from datetime import datetime, timezone
from typing import Dict, Set
from fastapi import WebSocket
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for broadcasting updates."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.user_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str = None):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(websocket)
        
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket, user_id: str = None):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
        
        if user_id and user_id in self.user_connections:
            self.user_connections[user_id].discard(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return
        
        message_text = json.dumps(message)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_text)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_to_user(self, user_id: str, message: dict):
        """Broadcast a message to all connections for a specific user."""
        if user_id not in self.user_connections:
            return
        
        message_text = json.dumps(message)
        disconnected = []
        
        for connection in self.user_connections[user_id]:
            try:
                await connection.send_text(message_text)
            except Exception as e:
                logger.error(f"Error broadcasting to user {user_id}: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection, user_id)
    
    async def broadcast_experiment_update(self, experiment_id: str, status: str, progress: dict = None):
        """Broadcast an experiment status update."""
        message = {
            "type": "experiment_update",
            "data": {
                "experiment_id": experiment_id,
                "status": status,
                "progress": progress or {},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        await self.broadcast(message)
    
    async def broadcast_new_round(self, experiment_id: str, round_num: int, round_data: dict):
        """Broadcast when a new round is completed."""
        message = {
            "type": "new_round",
            "data": {
                "experiment_id": experiment_id,
                "round_num": round_num,
                "round_summary": round_data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        await self.broadcast(message)


# Global connection manager instance
manager = ConnectionManager()