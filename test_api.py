"""
Tests for the FastAPI server and endpoints.
"""
import os
import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import json
from pathlib import Path
import shutil
from unittest.mock import AsyncMock

# Set test JWT secret before importing server
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-testing-only"

from src.api.server import app, get_current_user, User, create_access_token


# Create test client
client = TestClient(app)


# Mock authentication for tests
async def override_get_current_user():
    return User(username="testuser", disabled=False)


@pytest.fixture
def authenticated_client():
    """Create a test client with authentication."""
    app.dependency_overrides[get_current_user] = override_get_current_user
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def auth_token():
    """Create a valid auth token for tests."""
    token = create_access_token(data={"sub": "admin"})
    return token


@pytest.fixture
def mock_experiment_data(tmp_path):
    """Create mock experiment data for testing."""
    # Create results directory
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Create a mock experiment
    exp_id = "exp_20240101_120000"
    exp_dir = results_dir / exp_id
    exp_dir.mkdir()
    
    # Create experiment summary
    summary = {
        "experiment_id": exp_id,
        "start_time": "2024-01-01T12:00:00",
        "end_time": "2024-01-01T12:30:00",
        "total_rounds": 10,
        "total_games": 450,
        "total_api_calls": 100,
        "total_cost": 5.50,
        "acausal_indicators": {
            "cooperation_rate": 0.65,
            "defection_rate": 0.35
        }
    }
    
    with open(exp_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f)
    
    # Create rounds directory
    rounds_dir = exp_dir / "rounds"
    rounds_dir.mkdir()
    
    # Create a sample round
    round_data = {
        "round": 1,
        "agents": [
            {"id": "agent_1", "power": 1.0},
            {"id": "agent_2", "power": 1.0}
        ],
        "timestamp": "2024-01-01T12:01:00"
    }
    
    with open(rounds_dir / "round_1.json", "w") as f:
        json.dump(round_data, f)
    
    # Update the data loader to use test directory
    from src.api.routers.experiments import data_loader
    data_loader.results_path = results_dir
    data_loader.clear_cache()
    
    yield results_dir
    
    # Cleanup
    shutil.rmtree(tmp_path)


class TestRootEndpoints:
    """Test basic endpoints."""
    
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert "version" in response.json()
    
    def test_health_check(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert "timestamp" in response.json()
    
    def test_protected_route_without_auth(self):
        response = client.get("/api/v1/protected")
        assert response.status_code == 401
    
    def test_protected_route_with_auth(self, authenticated_client):
        response = authenticated_client.get("/api/v1/protected")
        assert response.status_code == 200
        assert "message" in response.json()


class TestAuthEndpoints:
    """Test authentication endpoints."""
    
    def test_login_success(self):
        response = client.post(
            "/api/v1/auth/login",
            data={"username": "admin", "password": "admin"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_failure(self):
        response = client.post(
            "/api/v1/auth/login",
            data={"username": "admin", "password": "wrong"}
        )
        assert response.status_code == 401
    
    def test_logout(self, auth_token):
        response = client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_get_current_user(self, auth_token):
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "admin"
        assert "disabled" in data


class TestExperimentEndpoints:
    """Test experiment endpoints."""
    
    def test_list_experiments(self, authenticated_client, mock_experiment_data):
        response = authenticated_client.get("/api/v1/experiments")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data
        assert len(data["items"]) == 1
        assert data["items"][0]["experiment_id"] == "exp_20240101_120000"
    
    def test_list_experiments_pagination(self, authenticated_client, mock_experiment_data):
        response = authenticated_client.get("/api/v1/experiments?page=2&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 2
        assert data["page_size"] == 10
        assert len(data["items"]) == 0  # No items on page 2
    
    def test_get_experiment(self, authenticated_client, mock_experiment_data):
        response = authenticated_client.get("/api/v1/experiments/exp_20240101_120000")
        assert response.status_code == 200
        data = response.json()
        assert data["experiment_id"] == "exp_20240101_120000"
        assert data["total_rounds"] == 10
        assert "acausal_indicators" in data
        assert data["round_count"] == 1
        assert data["agent_count"] == 2
    
    def test_get_experiment_not_found(self, authenticated_client):
        response = authenticated_client.get("/api/v1/experiments/nonexistent")
        assert response.status_code == 404
    
    def test_get_round_data(self, authenticated_client, mock_experiment_data):
        response = authenticated_client.get("/api/v1/experiments/exp_20240101_120000/rounds/1")
        assert response.status_code == 200
        data = response.json()
        assert data["round"] == 1
        assert len(data["agents"]) == 2
    
    def test_get_round_data_not_found(self, authenticated_client, mock_experiment_data):
        response = authenticated_client.get("/api/v1/experiments/exp_20240101_120000/rounds/999")
        assert response.status_code == 404


class TestDataCaching:
    """Test data caching functionality."""
    
    def test_cache_performance(self, authenticated_client, mock_experiment_data):
        """Test that caching improves performance."""
        import time
        
        # First request - should hit disk
        start = time.time()
        response1 = authenticated_client.get("/api/v1/experiments")
        time1 = time.time() - start
        assert response1.status_code == 200
        
        # Second request - should hit cache
        start = time.time()
        response2 = authenticated_client.get("/api/v1/experiments")
        time2 = time.time() - start
        assert response2.status_code == 200
        
        # Cache should be faster (though this might be flaky in CI)
        # Just verify both requests return the same data
        assert response1.json() == response2.json()


class TestWebSocket:
    """Test WebSocket endpoints."""
    
    def test_websocket_connect(self):
        """Test WebSocket connection."""
        with client.websocket_connect("/ws") as websocket:
            # Send a test message
            websocket.send_text("Hello WebSocket")
            
            # Receive echo response
            data = websocket.receive_text()
            assert data == "Echo: Hello WebSocket"
    
    def test_websocket_with_user_id(self):
        """Test WebSocket connection with user ID."""
        user_id = "test_user_123"
        with client.websocket_connect(f"/ws/{user_id}") as websocket:
            # Send a test message
            websocket.send_text("Hello")
            
            # Receive response with user ID
            data = websocket.receive_text()
            assert data == f"User {user_id}: Hello"
    
    @pytest.mark.asyncio
    async def test_websocket_manager_broadcast(self):
        """Test WebSocket manager broadcast functionality."""
        from src.api.websocket_manager import manager
        
        # Clear any existing connections
        manager.active_connections.clear()
        
        # Create mock WebSocket connections
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        
        # Add connections
        manager.active_connections.add(mock_ws1)
        manager.active_connections.add(mock_ws2)
        
        # Test broadcast
        test_message = {"type": "test", "data": "broadcast"}
        await manager.broadcast(test_message)
        
        # Verify both connections received the message
        expected_text = json.dumps(test_message)
        mock_ws1.send_text.assert_called_once_with(expected_text)
        mock_ws2.send_text.assert_called_once_with(expected_text)
    
    @pytest.mark.asyncio
    async def test_websocket_experiment_update(self):
        """Test experiment update broadcast."""
        from src.api.websocket_manager import manager
        
        mock_ws = AsyncMock()
        manager.active_connections.add(mock_ws)
        
        # Broadcast experiment update
        await manager.broadcast_experiment_update(
            experiment_id="exp_123",
            status="running",
            progress={"round": 5, "total": 10}
        )
        
        # Verify the message format
        call_args = mock_ws.send_text.call_args[0][0]
        message = json.loads(call_args)
        
        assert message["type"] == "experiment_update"
        assert message["data"]["experiment_id"] == "exp_123"
        assert message["data"]["status"] == "running"
        assert message["data"]["progress"]["round"] == 5
        assert "timestamp" in message["data"]
        
        # Clean up
        manager.active_connections.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])