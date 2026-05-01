"""
Tests for API endpoints.

Author: Inventions4All - github:TWeb79
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_service_info(self, client):
        """Test root endpoint returns service information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Stable Diffusion API Provider"
        assert data["version"] == "1.0.0"
        assert "docs" in data
        assert "health" in data
        assert "models" in data
        assert "generate" in data


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    @patch("src.api.routes.health.get_pipeline_manager")
    @patch("src.api.routes.health.get_device_info")
    def test_health_check(self, mock_device_info, mock_pipeline, client):
        """Test health check returns status."""
        mock_pipeline.return_value = MagicMock(
            current_model="test_model.safetensors"
        )
        # Create a proper DeviceInfo-like object
        mock_device_info.return_value = MagicMock()
        mock_device_info.return_value.name = "CPU"
        mock_device_info.return_value.type = "cpu"
        mock_device_info.return_value.mps_available = False
        mock_device_info.return_value.num_threads = 8
        mock_device_info.return_value.interop_threads = 4
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["loaded_model"] == "test_model.safetensors"
        assert data["device"] == "CPU"
        assert data["device_type"] == "cpu"
        assert data["mps_available"] is False
        assert data["torch_num_threads"] == 8


class TestModelsEndpoint:
    """Tests for models endpoint."""
    
    @patch("src.api.routes.models.get_pipeline_manager")
    def test_list_models(self, mock_pipeline, client):
        """Test listing available models."""
        mock_pipeline.return_value = MagicMock(
            discover_models=MagicMock(return_value=[
                {
                    "name": "model1.safetensors",
                    "path": "/models/model1.safetensors",
                    "size_bytes": 4000000000,
                    "size_mb": 3814.7,
                    "format": "safetensors",
                },
                {
                    "name": "model2.ckpt",
                    "path": "/models/model2.ckpt",
                    "size_bytes": 2000000000,
                    "size_mb": 1907.3,
                    "format": "ckpt",
                },
            ]),
            current_model="model1.safetensors",
        )
        
        response = client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["models"]) == 2
        assert data["current_model"] == "model1.safetensors"
    
    @patch("src.api.routes.models.get_pipeline_manager")
    def test_load_model_success(self, mock_pipeline, client):
        """Test loading a model successfully."""
        mock_pipeline.return_value = MagicMock(
            load_model=MagicMock(),
            current_model="test_model.safetensors",
            device="cpu",
            attention_slicing=True,
            cpu_offload=True,
        )
        
        response = client.post("/models/test_model.safetensors/load")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model"] == "test_model.safetensors"
        assert data["device"] == "cpu"
    
    @patch("src.api.routes.models.get_pipeline_manager")
    def test_load_model_not_found(self, mock_pipeline, client):
        """Test loading a non-existent model."""
        mock_pipeline.return_value = MagicMock(
            load_model=MagicMock(side_effect=FileNotFoundError("Model not found"))
        )
        
        response = client.post("/models/nonexistent/load")
        
        assert response.status_code == 404
    
    @patch("src.api.routes.models.get_pipeline_manager")
    def test_unload_model(self, mock_pipeline, client):
        """Test unloading a model."""
        mock_pipeline.return_value = MagicMock(unload=MagicMock())
        
        response = client.post("/models/unload")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


class TestGenerateEndpoint:
    """Tests for image generation endpoint."""
    
    def test_generate_requires_prompt(self):
        """Test that prompt is required."""
        with TestClient(app) as test_client:
            response = test_client.get("/generate")
            
            assert response.status_code == 422  # Validation error
    
    def test_invalid_dimensions(self):
        """Test that dimensions must be multiples of 8."""
        with TestClient(app) as test_client:
            response = test_client.get("/generate?prompt=test&width=500")
            
            assert response.status_code == 400
            assert "multiples of 8" in response.json()["detail"]
    
    @patch("src.api.routes.generate.get_pipeline_manager")
    def test_generate_without_loaded_model(self, mock_pipeline):
        """Test generation auto-loads model if not loaded."""
        mock_pm = MagicMock()
        mock_pm.is_loaded = False
        mock_pm.load_model = MagicMock()
        mock_pm.current_model = "test_model.safetensors"
        mock_pipeline.return_value = mock_pm
        
        # Mock the generate method to return a fake image
        mock_image = MagicMock()
        mock_image.save = MagicMock()
        mock_pm.generate = MagicMock(return_value=mock_image)
        
        with TestClient(app) as test_client:
            response = test_client.get("/generate?prompt=a cat")
        
        # Should attempt to load model first
        mock_pm.load_model.assert_called_once()


class TestSchemaValidation:
    """Tests for request/response schema validation."""
    
    def test_generate_request_validation(self):
        """Test GenerateRequest schema validation."""
        from src.schemas.generate import GenerateRequest
        
        # Valid request
        request = GenerateRequest(prompt="a cat")
        assert request.prompt == "a cat"
        assert request.steps == 25  # default
        assert request.width == 512  # default
        
        # Custom values
        request = GenerateRequest(
            prompt="a dog",
            steps=50,
            guidance=10.0,
            width=768,
            height=768,
            seed=42,
        )
        assert request.steps == 50
        assert request.guidance == 10.0
        assert request.width == 768
        assert request.seed == 42
    
    def test_generate_request_invalid_dimensions(self):
        """Test that invalid dimensions raise error."""
        from src.schemas.generate import GenerateRequest
        
        with pytest.raises(ValueError):
            GenerateRequest(prompt="test", width=500)  # not multiple of 8
    
    def test_model_info_schema(self):
        """Test ModelInfo schema."""
        from src.schemas.model import ModelInfo
        
        model = ModelInfo(
            name="test.safetensors",
            path="/models/test.safetensors",
            size_bytes=4000000000,
            size_mb=3814.7,
            format="safetensors",
        )
        
        assert model.name == "test.safetensors"
        assert model.format == "safetensors"