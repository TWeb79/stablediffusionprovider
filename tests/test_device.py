"""
Tests for device detection and management.

Author: Inventions4All - github:TWeb79
"""

import pytest
from unittest.mock import patch, MagicMock

from src.core.device import (
    DeviceInfo,
    detect_device,
    get_device_info,
    get_memory_requirements,
    optimize_for_device,
)


class TestDetectDevice:
    """Tests for device detection."""
    
    @patch("torch.backends.mps.is_available")
    def test_detect_auto_prefers_mps(self, mock_mps):
        """Auto detection should prefer MPS when available."""
        mock_mps.return_value = True
        
        device = detect_device()
        assert device == "mps"
    
    @patch("torch.backends.mps.is_available")
    def test_detect_auto_falls_back_to_cpu(self, mock_mps):
        """Auto detection should fall back to CPU when MPS unavailable."""
        mock_mps.return_value = False
        
        device = detect_device()
        assert device == "cpu"
    
    @patch("torch.backends.mps.is_available")
    def test_prefer_mps(self, mock_mps):
        """Explicit MPS preference uses MPS when available."""
        mock_mps.return_value = True
        
        device = detect_device("mps")
        assert device == "mps"
    
    @patch("torch.backends.mps.is_available")
    def test_mps_requested_but_unavailable(self, mock_mps):
        """MPS request falls back to CPU when unavailable."""
        mock_mps.return_value = False
        
        device = detect_device("mps")
        assert device == "cpu"


class TestGetDeviceInfo:
    """Tests for getting device information."""
    
    @patch("torch.backends.mps.is_available")
    @patch("torch.get_num_interop_threads")
    @patch("torch.get_num_threads")
    def test_mps_device_info(self, mock_threads, mock_interop, mock_mps):
        """Test getting MPS device info."""
        mock_mps.return_value = True
        mock_threads.return_value = 8
        mock_interop.return_value = 4
        
        info = get_device_info("mps")
        
        assert info.name == "Apple Metal (MPS)"
        assert info.type == "mps"
        assert info.mps_available is True
        assert info.num_threads == 8
        assert info.interop_threads == 4
    
    @patch("torch.backends.mps.is_available")
    @patch("torch.get_num_interop_threads")
    @patch("torch.get_num_threads")
    def test_cpu_device_info(self, mock_threads, mock_interop, mock_mps):
        """Test getting CPU device info."""
        mock_mps.return_value = False
        mock_threads.return_value = 16
        mock_interop.return_value = 8
        
        info = get_device_info("cpu")
        
        assert info.name == "CPU"
        assert info.type == "cpu"
        assert info.mps_available is False
        assert info.num_threads == 16


class TestGetMemoryRequirements:
    """Tests for memory requirement estimation."""
    
    def test_standard_resolution(self):
        """Test memory requirements for standard resolution."""
        result = get_memory_requirements(512, 512)
        
        assert "model_gb" in result
        assert "inference_gb" in result
        assert "total_gb" in result
        assert "recommended_gb" in result
        
        # Model should be 4GB
        assert result["model_gb"] == 4.0
        
        # Inference should scale with resolution
        assert result["inference_gb"] == 2.0  # 512x512 is base resolution
    
    def test_higher_resolution(self):
        """Test memory requirements for higher resolution."""
        result_512 = get_memory_requirements(512, 512)
        result_1024 = get_memory_requirements(1024, 1024)
        
        # Higher resolution should require more memory
        assert result_1024["inference_gb"] > result_512["inference_gb"]
    
    def test_custom_model_size(self):
        """Test memory requirements with custom model size."""
        result = get_memory_requirements(512, 512, model_size_gb=8.0)
        
        assert result["model_gb"] == 8.0


class TestOptimizeForDevice:
    """Tests for device optimization settings."""
    
    def test_cpu_optimizations(self):
        """Test optimization settings for CPU."""
        settings = optimize_for_device("cpu", attention_slicing=True, cpu_offload=True)
        
        assert settings["device"] == "cpu"
        # CPU always uses attention slicing
        assert settings["attention_slicing"] is True
        assert settings["cpu_offload"] is True
        assert settings["enable_vae_slicing"] is True
        assert settings["enable_sequential_cpu_offload"] is True
    
    def test_mps_optimizations(self):
        """Test optimization settings for MPS."""
        settings = optimize_for_device("mps", attention_slicing=False, cpu_offload=False)
        
        assert settings["device"] == "mps"
        assert settings["attention_slicing"] is False
        assert settings["cpu_offload"] is False