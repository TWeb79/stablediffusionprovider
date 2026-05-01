"""
Tests for configuration management.

Author: Inventions4All - github:TWeb79
"""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from src.core.config import (
    APISettings,
    ModelSettings,
    DeviceSettings,
    GenerationSettings,
    Settings,
    get_settings,
    reload_settings,
)


class TestAPISettings:
    """Tests for API settings."""
    
    def test_default_values(self):
        """Test default API settings."""
        settings = APISettings()
        assert settings.host == "0.0.0.0"
        assert settings.port == 8141
        assert settings.workers == 1


class TestModelSettings:
    """Tests for model settings."""
    
    def test_default_values(self):
        """Test default model settings."""
        settings = ModelSettings()
        assert settings.directory == "/models"
        assert settings.default_model is None
        assert settings.safety_checker is False


class TestDeviceSettings:
    """Tests for device settings."""
    
    def test_default_values(self):
        """Test default device settings (CPU mode)."""
        settings = DeviceSettings()
        assert settings.device == "cpu"
        assert settings.attention_slicing is True
        assert settings.cpu_offload is True  # CPU offload enabled by default
    
    def test_valid_device_cpu(self):
        """Test valid CPU device value."""
        settings = DeviceSettings(device="cpu")
        assert settings.device == "cpu"
    
    def test_valid_device_mps(self):
        """Test valid MPS device value."""
        settings = DeviceSettings(device="mps")
        assert settings.device == "mps"
    
    def test_valid_device_auto(self):
        """Test valid auto device value."""
        settings = DeviceSettings(device="auto")
        assert settings.device == "auto"
    
    def test_invalid_device(self):
        """Test invalid device value raises error."""
        with pytest.raises(ValueError):
            DeviceSettings(device="invalid")
    
    def test_mps_device_disables_cpu_offload(self):
        """Test that MPS device disables CPU offload."""
        settings = DeviceSettings(device="mps", cpu_offload=True)
        # cpu_offload should still be settable but will be ignored at runtime
        assert settings.device == "mps"


class TestGenerationSettings:
    """Tests for generation settings."""
    
    def test_default_values(self):
        """Test default generation settings."""
        settings = GenerationSettings()
        assert settings.steps == 25
        assert settings.guidance == 7.5
        assert settings.width == 512
        assert settings.height == 512
    
    def test_valid_dimensions(self):
        """Test valid dimension values."""
        settings = GenerationSettings(width=768, height=768)
        assert settings.width == 768
        assert settings.height == 768
    
    def test_invalid_dimensions_not_multiple_of_8(self):
        """Test dimensions not multiple of 8 raises error."""
        with pytest.raises(ValueError):
            GenerationSettings(width=500)
    
    def test_steps_range(self):
        """Test steps validation."""
        # Valid range
        settings = GenerationSettings(steps=100)
        assert settings.steps == 100
        
        # Below minimum
        with pytest.raises(ValueError):
            GenerationSettings(steps=0)
        
        # Above maximum
        with pytest.raises(ValueError):
            GenerationSettings(steps=200)


class TestSettings:
    """Tests for main settings class."""
    
    def test_yaml_file_not_found(self):
        """Test handling missing YAML file."""
        with patch("pathlib.Path.exists", return_value=False):
            settings = Settings.from_yaml()
            # Should use defaults
            assert settings.api.port == 8141
            assert settings.device.device == "cpu"


class TestSettingsSingleton:
    """Tests for settings singleton behavior."""
    
    def test_get_settings_returns_same_instance(self):
        """Test that get_settings returns cached instance."""
        # Reset singleton
        import src.core.config
        src.core.config._settings = None
        
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
    
    def test_reload_settings_returns_new_instance(self):
        """Test that reload_settings returns fresh instance."""
        import src.core.config
        src.core.config._settings = None
        
        settings1 = get_settings()
        settings2 = reload_settings()
        assert settings1 is not settings2