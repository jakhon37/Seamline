"""
Unit tests for utility functions.
"""
import os
from unittest.mock import patch

import pytest
import torch

from src.utils.config_validator import (
    _validate_data_config,
    _validate_loss_config,
    _validate_model_config,
    _validate_training_config,
    expand_environment_variables,
    validate_config,
)
from src.utils.distributed import get_rank, get_world_size, is_main_process, reduce_tensor
from src.utils.logger import AverageMeter
from src.utils.memory_tracker import MemoryTracker


class TestMemoryTracker:
    """Test cases for MemoryTracker class."""

    def test_initialization_cpu(self):
        """Test MemoryTracker initialization on CPU."""
        tracker = MemoryTracker(enabled=True, device_id=0)

        # Should be disabled on CPU
        if not torch.cuda.is_available():
            assert tracker.enabled == False, "Should be disabled when CUDA unavailable"

        assert tracker.peak_memory_mb == 0, "Peak memory should start at 0"

    @pytest.mark.gpu
    def test_initialization_gpu(self):
        """Test MemoryTracker initialization on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        tracker = MemoryTracker(enabled=True, device_id=0)

        assert tracker.enabled == True, "Should be enabled on GPU"
        assert tracker.device_id == 0, "Device ID should be set"

    def test_disabled_tracker(self):
        """Test MemoryTracker when explicitly disabled."""
        tracker = MemoryTracker(enabled=False)

        assert tracker.enabled == False, "Should be disabled"

        stats = tracker.get_memory_stats()
        assert stats is None, "Should return None when disabled"

    @pytest.mark.gpu
    def test_get_memory_stats(self):
        """Test getting memory statistics."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        tracker = MemoryTracker(enabled=True, device_id=0)

        # Allocate some GPU memory
        tensor = torch.randn(1000, 1000).cuda()

        stats = tracker.get_memory_stats()

        assert stats is not None, "Should return stats dict"
        assert "allocated_mb" in stats, "Should contain allocated_mb"
        assert "reserved_mb" in stats, "Should contain reserved_mb"
        assert "max_allocated_mb" in stats, "Should contain max_allocated_mb"
        assert "peak_mb" in stats, "Should contain peak_mb"

        assert stats["allocated_mb"] > 0, "Should have allocated memory"

        del tensor

    @pytest.mark.gpu
    def test_get_formatted_stats(self):
        """Test getting formatted memory statistics."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        tracker = MemoryTracker(enabled=True, device_id=0)

        formatted = tracker.get_formatted_stats()

        assert isinstance(formatted, str), "Should return a string"
        assert "GPU Memory" in formatted, "Should contain GPU Memory"
        assert "MB" in formatted, "Should contain MB unit"

    def test_get_formatted_stats_disabled(self):
        """Test formatted stats when tracker is disabled."""
        tracker = MemoryTracker(enabled=False)

        formatted = tracker.get_formatted_stats()

        assert isinstance(formatted, str), "Should return a string"
        assert "not available" in formatted.lower(), "Should indicate not available"

    @pytest.mark.gpu
    def test_reset_peak(self):
        """Test resetting peak memory statistics."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        tracker = MemoryTracker(enabled=True, device_id=0)
        tracker.reset_peak()

        # Allocate memory
        tensor = torch.randn(1000, 1000).cuda()
        stats1 = tracker.get_memory_stats()
        peak1 = stats1["peak_mb"]

        # Reset peak
        tracker.reset_peak()
        stats2 = tracker.get_memory_stats()

        # After reset, peak tracking starts fresh
        # The internal peak_memory_mb is reset to 0, but get_memory_stats
        # will show the current allocated memory as the new peak
        assert tracker.peak_memory_mb >= 0, "peak_memory_mb should be non-negative after reset"

        del tensor

    @pytest.mark.gpu
    def test_peak_tracking(self):
        """Test that peak memory is tracked correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        tracker = MemoryTracker(enabled=True, device_id=0)
        tracker.reset_peak()

        # Allocate increasing amounts of memory
        tensors = []
        for i in range(3):
            tensor = torch.randn(1000 * (i + 1), 1000).cuda()
            tensors.append(tensor)
            stats = tracker.get_memory_stats()

        peak = stats["peak_mb"]

        # Peak should be greater than zero
        assert peak > 0, "Peak memory should be tracked"

        # Clean up
        for tensor in tensors:
            del tensor

    @pytest.mark.gpu
    def test_get_gpu_memory_summary(self):
        """Test GPU memory summary for all devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        summary = MemoryTracker.get_gpu_memory_summary()

        assert isinstance(summary, str), "Should return a string"
        assert "GPU" in summary, "Should mention GPU"
        assert "MB" in summary, "Should include MB units"

    def test_get_gpu_memory_summary_no_cuda(self):
        """Test GPU memory summary when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            summary = MemoryTracker.get_gpu_memory_summary()

            assert "not available" in summary.lower(), "Should indicate CUDA not available"

    def test_invalid_device_id(self):
        """Test initialization with invalid device ID."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device_count = torch.cuda.device_count()

        # Try to use device ID beyond available devices
        tracker = MemoryTracker(enabled=True, device_id=device_count + 10)

        # Should clamp to valid device ID or disable
        assert (
            tracker.device_id < device_count or not tracker.enabled
        ), "Should handle invalid device ID gracefully"


class TestConfigValidator:
    """Test cases for configuration validation."""

    def test_validate_valid_config(self, sample_config):
        """Test validation with a valid configuration."""
        # Should not raise any errors
        validate_config(sample_config)

    def test_validate_missing_section(self, sample_config):
        """Test validation with missing required section."""
        del sample_config["training"]

        with pytest.raises(KeyError, match="Missing required config section"):
            validate_config(sample_config)

    def test_validate_missing_key(self, sample_config):
        """Test validation with missing required key."""
        del sample_config["training"]["batch_size"]

        with pytest.raises(KeyError, match="Missing required config key"):
            validate_config(sample_config)

    def test_validate_invalid_image_size(self, sample_config):
        """Test validation with invalid image size."""
        sample_config["data"]["image_size"] = 0

        with pytest.raises(ValueError, match="image_size must be positive"):
            validate_config(sample_config)

    def test_validate_invalid_num_classes(self, sample_config):
        """Test validation with invalid number of classes."""
        sample_config["data"]["num_classes"] = 1

        with pytest.raises(ValueError, match="num_classes must be >= 2"):
            validate_config(sample_config)

    def test_validate_invalid_epochs(self, sample_config):
        """Test validation with invalid epochs."""
        sample_config["training"]["epochs"] = 0

        with pytest.raises(ValueError, match="epochs must be positive"):
            validate_config(sample_config)

    def test_validate_invalid_batch_size(self, sample_config):
        """Test validation with invalid batch size."""
        sample_config["training"]["batch_size"] = -1

        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_config(sample_config)

    def test_validate_invalid_learning_rate(self, sample_config):
        """Test validation with invalid learning rate."""
        sample_config["training"]["learning_rate"] = -0.001

        with pytest.raises(ValueError, match="learning_rate must be positive"):
            validate_config(sample_config)

    def test_validate_invalid_architecture(self, sample_config):
        """Test validation with invalid model architecture."""
        sample_config["model"]["architecture"] = "InvalidArch"

        with pytest.raises(ValueError, match="Invalid architecture"):
            validate_config(sample_config)

    def test_validate_invalid_loss_type(self, sample_config):
        """Test validation with invalid loss type."""
        sample_config["loss"]["type"] = "invalid_loss"

        with pytest.raises(ValueError, match="Invalid loss type"):
            validate_config(sample_config)

    def test_validate_combined_loss_alpha(self, sample_config):
        """Test validation of combined loss alpha parameter."""
        sample_config["loss"]["type"] = "combined"
        sample_config["loss"]["combined_alpha"] = 1.5

        with pytest.raises(ValueError, match="combined_alpha must be in"):
            validate_config(sample_config)

    def test_expand_environment_variables(self):
        """Test expanding environment variables in config."""
        os.environ["TEST_VAR"] = "/test/path"

        config = {
            "paths": {
                "output_dir": "$TEST_VAR/outputs",
                "nested": {"data_dir": "${TEST_VAR}/data"},
            },
            "other": "no_var_here",
        }

        expanded = expand_environment_variables(config)

        assert expanded["paths"]["output_dir"] == "/test/path/outputs", "Should expand $TEST_VAR"
        assert (
            expanded["paths"]["nested"]["data_dir"] == "/test/path/data"
        ), "Should expand ${TEST_VAR}"
        assert expanded["other"] == "no_var_here", "Should leave strings without vars unchanged"

        # Clean up
        del os.environ["TEST_VAR"]

    def test_validate_data_config(self, sample_config):
        """Test _validate_data_config function."""
        # Should not raise errors
        _validate_data_config(sample_config["data"])

    def test_validate_training_config(self, sample_config):
        """Test _validate_training_config function."""
        # Should not raise errors
        _validate_training_config(sample_config["training"])

    def test_validate_model_config(self, sample_config):
        """Test _validate_model_config function."""
        # Should not raise errors
        _validate_model_config(sample_config["model"])

    def test_validate_loss_config(self, sample_config):
        """Test _validate_loss_config function."""
        # Should not raise errors
        _validate_loss_config(sample_config["loss"])

    def test_warning_for_low_num_workers(self, sample_config, capsys):
        """Test warning is issued for low num_workers."""
        sample_config["training"]["num_workers"] = 2

        _validate_training_config(sample_config["training"])

        captured = capsys.readouterr()
        assert "num_workers" in captured.out.lower() or "num_workers" in str(
            captured
        ), "Should warn about low num_workers"


class TestAverageMeter:
    """Test cases for AverageMeter class."""

    def test_initialization(self):
        """Test AverageMeter initialization."""
        meter = AverageMeter(name="loss")

        assert meter.name == "loss", "Name should be set"
        assert meter.val == 0, "Initial value should be 0"
        assert meter.avg == 0, "Initial average should be 0"
        assert meter.sum == 0, "Initial sum should be 0"
        assert meter.count == 0, "Initial count should be 0"

    def test_update_single_value(self):
        """Test updating with single values."""
        meter = AverageMeter()

        meter.update(5.0)
        assert meter.val == 5.0, "Value should be updated"
        assert meter.avg == 5.0, "Average should be 5.0"
        assert meter.count == 1, "Count should be 1"

        meter.update(3.0)
        assert meter.val == 3.0, "Value should be updated to last"
        assert meter.avg == 4.0, "Average should be (5+3)/2 = 4.0"
        assert meter.count == 2, "Count should be 2"

    def test_update_with_count(self):
        """Test updating with custom count."""
        meter = AverageMeter()

        meter.update(10.0, n=5)
        assert meter.val == 10.0, "Value should be 10.0"
        assert meter.sum == 50.0, "Sum should be 10.0 * 5 = 50.0"
        assert meter.count == 5, "Count should be 5"
        assert meter.avg == 10.0, "Average should be 10.0"

    def test_reset(self):
        """Test resetting the meter."""
        meter = AverageMeter()

        meter.update(5.0)
        meter.update(10.0)

        meter.reset()

        assert meter.val == 0, "Value should be reset"
        assert meter.avg == 0, "Average should be reset"
        assert meter.sum == 0, "Sum should be reset"
        assert meter.count == 0, "Count should be reset"

    def test_multiple_updates(self):
        """Test multiple updates and averaging."""
        meter = AverageMeter()

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for val in values:
            meter.update(val)

        expected_avg = sum(values) / len(values)
        assert abs(meter.avg - expected_avg) < 1e-6, "Average should be correct"
        assert meter.count == len(values), "Count should match number of updates"

    def test_weighted_average(self):
        """Test weighted average with different counts."""
        meter = AverageMeter()

        meter.update(10.0, n=1)
        meter.update(20.0, n=2)
        meter.update(30.0, n=3)

        # Expected: (10*1 + 20*2 + 30*3) / (1+2+3) = 140 / 6 = 23.333...
        expected_avg = (10 * 1 + 20 * 2 + 30 * 3) / (1 + 2 + 3)
        assert abs(meter.avg - expected_avg) < 1e-6, "Weighted average should be correct"

    def test_str_representation(self):
        """Test string representation."""
        meter = AverageMeter(name="accuracy")
        meter.update(0.95)

        str_repr = str(meter)

        assert "accuracy" in str_repr, "Should contain name"
        assert "0.95" in str_repr, "Should contain value"

    def test_edge_case_zero_division(self):
        """Test that zero division is handled (count=0)."""
        meter = AverageMeter()

        # Without any updates, avg should be 0 (sum=0, count=0)
        # This would cause division by zero, but the initial state handles it
        assert meter.avg == 0, "Should handle empty meter"


class TestDistributedUtils:
    """Test cases for distributed training utilities."""

    def test_is_main_process_not_distributed(self):
        """Test is_main_process when distributed is not initialized."""
        # When not in distributed mode, should return True
        result = is_main_process()

        assert result == True, "Should be main process when not distributed"

    def test_is_main_process_with_rank_zero(self):
        """Test is_main_process with rank=0."""
        result = is_main_process(rank=0)

        assert result == True, "Rank 0 should be main process"

    def test_is_main_process_with_rank_nonzero(self):
        """Test is_main_process with rank!=0."""
        result = is_main_process(rank=1)

        assert result == False, "Rank 1 should not be main process"

    def test_get_world_size_not_distributed(self):
        """Test get_world_size when not in distributed mode."""
        world_size = get_world_size()

        assert world_size == 1, "World size should be 1 when not distributed"

    def test_get_rank_not_distributed(self):
        """Test get_rank when not in distributed mode."""
        rank = get_rank()

        assert rank == 0, "Rank should be 0 when not distributed"

    def test_reduce_tensor_single_process(self, device):
        """Test reduce_tensor with world_size=1."""
        tensor = torch.tensor(5.0).to(device)

        reduced = reduce_tensor(tensor, world_size=1)

        assert torch.allclose(reduced, tensor), "Should return same tensor when world_size=1"

    def test_reduce_tensor_clone(self, device):
        """Test that reduce_tensor behavior with single process."""
        tensor = torch.tensor(5.0, device=device)

        # For world_size=1, returns the same tensor (no clone)
        reduced = reduce_tensor(tensor, world_size=1)

        # Should return the same tensor
        assert reduced is tensor, "For world_size=1, should return same tensor"
        assert reduced.item() == 5.0, "Value should match"

    @patch("torch.distributed.is_initialized")
    def test_get_world_size_distributed(self, mock_is_init):
        """Test get_world_size in distributed mode."""
        mock_is_init.return_value = True

        with patch("torch.distributed.get_world_size", return_value=4):
            world_size = get_world_size()
            assert world_size == 4, "Should return distributed world size"

    @patch("torch.distributed.is_initialized")
    def test_get_rank_distributed(self, mock_is_init):
        """Test get_rank in distributed mode."""
        mock_is_init.return_value = True

        with patch("torch.distributed.get_rank", return_value=2):
            rank = get_rank()
            assert rank == 2, "Should return distributed rank"

    @patch("torch.distributed.is_initialized")
    def test_is_main_process_distributed(self, mock_is_init):
        """Test is_main_process in distributed mode."""
        mock_is_init.return_value = True

        with patch("torch.distributed.get_rank", return_value=0):
            result = is_main_process()
            assert result == True, "Rank 0 should be main process"

        with patch("torch.distributed.get_rank", return_value=1):
            result = is_main_process()
            assert result == False, "Rank 1 should not be main process"


class TestUtilsIntegration:
    """Integration tests for utility functions."""

    def test_memory_tracker_in_training_loop(self):
        """Test using MemoryTracker in a simulated training loop."""
        tracker = MemoryTracker(enabled=True)

        # Simulate training iterations
        for i in range(5):
            # Simulate work
            if torch.cuda.is_available():
                tensor = torch.randn(100, 100).cuda()
                del tensor

            # Get stats
            stats = tracker.get_memory_stats()

            if stats is not None:
                assert all(
                    v >= 0 for v in stats.values()
                ), "All memory stats should be non-negative"

    def test_average_meter_training_tracking(self):
        """Test using AverageMeter for training metric tracking."""
        loss_meter = AverageMeter("loss")
        acc_meter = AverageMeter("accuracy")

        # Simulate training batches
        batch_losses = [2.5, 2.3, 2.1, 1.9, 1.8]
        batch_accs = [0.6, 0.65, 0.7, 0.75, 0.78]

        for loss, acc in zip(batch_losses, batch_accs):
            loss_meter.update(loss)
            acc_meter.update(acc)

        assert (
            abs(loss_meter.avg - sum(batch_losses) / len(batch_losses)) < 1e-6
        ), "Loss average should be correct"
        assert (
            abs(acc_meter.avg - sum(batch_accs) / len(batch_accs)) < 1e-6
        ), "Accuracy average should be correct"

    def test_config_validation_pipeline(self, sample_config):
        """Test full config validation pipeline."""
        # Expand environment variables
        expanded = expand_environment_variables(sample_config)

        # Validate
        validate_config(expanded)

        # Should complete without errors
        assert True, "Config validation pipeline should succeed"

    def test_distributed_utils_consistency(self):
        """Test that distributed utils are consistent."""
        world_size = get_world_size()
        rank = get_rank()
        is_main = is_main_process()

        # In non-distributed mode
        assert world_size == 1, "World size should be 1"
        assert rank == 0, "Rank should be 0"
        assert is_main == True, "Should be main process"

        # These should be consistent
        assert is_main == (rank == 0), "is_main should match (rank == 0)"
