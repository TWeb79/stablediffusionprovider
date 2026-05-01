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
    
    @patch("torch.cuda.is_available")
    def test_detect_cuda_available(self, mock_cuda_available):
        """Test detection when CUDA is available."""
        mock_cuda_available.return_value = True
        
        device = detect_device()
        assert device == "cuda"
    
    @patch("torch.cuda.is_available")
    def test_detect_cuda_not_available(self, mock_cuda_available):
        """Test detection when CUDA is not available."""
        mock_cuda_available.return_value = False
        
        device = detect_device()
        assert device == "cpu"
    
    @patch("torch.cuda.is_available")
    def test_prefer_cuda(self, mock_cuda_available):
        """Test explicit CUDA preference."""
        mock_cuda_available.return_value = True
        
        device = detect_device("cuda")
        assert device == "cuda"
    
    @patch("torch.cuda.is_available")
    def test_prefer_cpu(self, mock_cuda_available):
        """Test explicit CPU preference."""
        mock_cuda_available.return_value = True
        
        device = detect_device("cpu")
        assert device == "cpu"
    
    @patch("torch.cuda.is_available")
    def test_cuda_requested_but_unavailable(self, mock_cuda_available):
        """Test CUDA requested but not available falls back to CPU."""
        mock_cuda_available.return_value = False
        
        device = detect_device("cuda")
        assert device == "cpu"


class TestGetDeviceInfo:
    """Tests for getting device information."""
    
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_name")
    @patch("torch.cuda.get_device_properties")
    def test_cuda_device_info(self, mock_props, mock_name, mock_cuda_available):
        """Test getting CUDA device info."""
        mock_cuda_available.return_value = True
        mock_name.return_value = "NVIDIA RTX 3090"
        mock_props.return_value = MagicMock(total_memory=24 * 1024**3)
        
        info = get_device_info("cuda")
        
        assert info.name == "NVIDIA RTX 3090"
        assert info.type == "cuda"
        assert info.cuda_available is True
        assert info.cuda_device_count == 1
    
    @patch("torch.cuda.is_available")
    def test_cpu_device_info(self, mock_cuda_available):
        """Test getting CPU device info."""
        mock_cuda_available.return_value = False
        
        info = get_device_info("cpu")
        
        assert info.name == "CPU"
        assert info.type == "cpu"
        assert info.cuda_available is False


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
    
    def test_cuda_optimizations(self):
        """Test optimization settings for CUDA."""
        settings = optimize_for_device("cuda", attention_slicing=True, cpu_offload=True)
        
        assert settings["device"] == "cuda"
        assert settings["attention_slicing"] is True
        assert settings["cpu_offload"] is True
        assert settings["enable_vae_slicing"] is True
        assert settings["enable_sequential_cpu_offload"] is True
    
    def test_cpu_optimizations(self):
        """Test optimization settings for CPU."""
        settings = optimize_for_device("cpu", attention_slicing=True, cpu_offload=True)
        
        assert settings["device"] == "cpu"
        # CPU always uses attention slicing
        assert settings["attention_slicing"] is True
        # CPU offload doesn't apply to CPU device
        assert settings["cpu_offload"] is False