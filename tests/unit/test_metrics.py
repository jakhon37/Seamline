"""
Unit tests for metrics.
"""
import pytest
import torch
import numpy as np

from src.utils.metrics import SegmentationMetrics, dice_coefficient


class TestSegmentationMetrics:
    """Test cases for SegmentationMetrics class."""
    
    def test_initialization(self, device):
        """Test SegmentationMetrics initialization."""
        metrics = SegmentationMetrics(
            num_classes=2,
            threshold=0.5,
            device=device,
            compute_on_cpu=True
        )
        
        assert metrics.num_classes == 2, "num_classes should be set correctly"
        assert metrics.threshold == 0.5, "threshold should be set correctly"
        assert metrics.compute_on_cpu == True, "compute_on_cpu should be set"
        assert metrics.device == torch.device('cpu'), "Should use CPU when compute_on_cpu=True"
    
    def test_initialization_without_cpu_compute(self, device):
        """Test initialization with compute_on_cpu=False."""
        metrics = SegmentationMetrics(
            num_classes=2,
            device=device,
            compute_on_cpu=False
        )
        
        assert metrics.device == device, "Should use provided device"
    
    def test_update_with_class_indices(self, device):
        """Test updating metrics with class indices."""
        metrics = SegmentationMetrics(num_classes=2, compute_on_cpu=True)
        
        # Create predictions and targets as class indices
        batch_size, H, W = 2, 16, 16
        preds = torch.randint(0, 2, (batch_size, H, W)).to(device)
        targets = torch.randint(0, 2, (batch_size, H, W)).to(device)
        
        # Should not raise any errors
        metrics.update(preds, targets)
    
    def test_update_with_logits(self, device):
        """Test updating metrics with logits."""
        metrics = SegmentationMetrics(num_classes=2, compute_on_cpu=True)
        
        # Create predictions as logits [B, C, H, W]
        batch_size, num_classes, H, W = 2, 2, 16, 16
        preds = torch.randn(batch_size, num_classes, H, W).to(device)
        targets = torch.randint(0, num_classes, (batch_size, H, W)).to(device)
        
        # Should convert logits to class indices automatically
        metrics.update(preds, targets)
    
    def test_update_with_onehot_targets(self, device):
        """Test updating metrics with one-hot encoded targets."""
        metrics = SegmentationMetrics(num_classes=2, compute_on_cpu=True)
        
        batch_size, num_classes, H, W = 2, 2, 16, 16
        preds = torch.randn(batch_size, num_classes, H, W).to(device)
        
        # Create one-hot targets
        targets_indices = torch.randint(0, num_classes, (batch_size, H, W))
        targets_onehot = torch.zeros(batch_size, num_classes, H, W)
        targets_onehot.scatter_(1, targets_indices.unsqueeze(1), 1)
        targets_onehot = targets_onehot.to(device)
        
        # Should convert one-hot to class indices automatically
        metrics.update(preds, targets_onehot)
    
    def test_compute_returns_all_metrics(self, device):
        """Test that compute returns all expected metrics."""
        metrics = SegmentationMetrics(num_classes=2, compute_on_cpu=True)
        
        # Add some data
        preds = torch.randint(0, 2, (2, 16, 16)).to(device)
        targets = torch.randint(0, 2, (2, 16, 16)).to(device)
        metrics.update(preds, targets)
        
        result = metrics.compute()
        
        assert isinstance(result, dict), "Should return a dictionary"
        assert "f1" in result, "Should contain F1 score"
        assert "precision" in result, "Should contain precision"
        assert "recall" in result, "Should contain recall"
        assert "iou" in result, "Should contain IoU"
        
        # Check that all values are scalars
        for key, value in result.items():
            assert isinstance(value, float), f"{key} should be a float"
            assert 0 <= value <= 1, f"{key} should be in [0, 1]"
    
    def test_reset_clears_metrics(self, device):
        """Test that reset clears accumulated metrics."""
        metrics = SegmentationMetrics(num_classes=2, compute_on_cpu=True)
        
        # Add some data
        preds = torch.randint(0, 2, (2, 16, 16)).to(device)
        targets = torch.randint(0, 2, (2, 16, 16)).to(device)
        metrics.update(preds, targets)
        
        # Compute metrics
        result1 = metrics.compute()
        
        # Reset
        metrics.reset()
        
        # Add different data
        preds2 = torch.ones(2, 16, 16, dtype=torch.long).to(device)
        targets2 = torch.ones(2, 16, 16, dtype=torch.long).to(device)
        metrics.update(preds2, targets2)
        
        result2 = metrics.compute()
        
        # Results should be different (unless by coincidence)
        # For perfect prediction, F1 should be 1.0
        assert result2["f1"] == 1.0, "Perfect prediction should give F1=1.0"
    
    def test_perfect_prediction_metrics(self, device):
        """Test metrics with perfect predictions."""
        metrics = SegmentationMetrics(num_classes=2, compute_on_cpu=True)
        
        # Create identical predictions and targets
        preds = torch.randint(0, 2, (2, 16, 16)).to(device)
        targets = preds.clone()
        
        metrics.update(preds, targets)
        result = metrics.compute()
        
        # All metrics should be 1.0 for perfect predictions
        assert result["f1"] == 1.0, "F1 should be 1.0 for perfect predictions"
        assert result["precision"] == 1.0, "Precision should be 1.0"
        assert result["recall"] == 1.0, "Recall should be 1.0"
        assert result["iou"] == 1.0, "IoU should be 1.0"
    
    def test_worst_prediction_metrics(self, device):
        """Test metrics with completely wrong predictions."""
        metrics = SegmentationMetrics(num_classes=2, compute_on_cpu=True)
        
        # Create opposite predictions
        targets = torch.zeros(2, 16, 16, dtype=torch.long).to(device)
        preds = torch.ones(2, 16, 16, dtype=torch.long).to(device)
        
        metrics.update(preds, targets)
        result = metrics.compute()
        
        # Metrics should be very low (but exact values depend on averaging)
        assert result["f1"] < 0.5, "F1 should be low for wrong predictions"
        assert result["iou"] < 0.5, "IoU should be low for wrong predictions"
    
    def test_multiple_updates(self, device):
        """Test accumulating metrics over multiple updates."""
        metrics = SegmentationMetrics(num_classes=2, compute_on_cpu=True)
        
        # Add data in multiple batches
        for _ in range(5):
            preds = torch.randint(0, 2, (2, 16, 16)).to(device)
            targets = torch.randint(0, 2, (2, 16, 16)).to(device)
            metrics.update(preds, targets)
        
        result = metrics.compute()
        
        # Should compute metrics across all batches
        assert all(0 <= v <= 1 for v in result.values()), "All metrics should be in [0, 1]"
    
    def test_multiclass_metrics(self, device):
        """Test metrics with more than 2 classes."""
        num_classes = 5
        metrics = SegmentationMetrics(num_classes=num_classes, compute_on_cpu=True)
        
        preds = torch.randint(0, num_classes, (2, 16, 16)).to(device)
        targets = torch.randint(0, num_classes, (2, 16, 16)).to(device)
        
        metrics.update(preds, targets)
        result = metrics.compute()
        
        assert all(0 <= v <= 1 for v in result.values()), "All metrics should be in [0, 1]"
    
    def test_compute_on_cpu_saves_gpu_memory(self):
        """Test that compute_on_cpu actually moves tensors to CPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        metrics = SegmentationMetrics(num_classes=2, device=device, compute_on_cpu=True)
        
        # Create GPU tensors
        preds = torch.randint(0, 2, (2, 16, 16)).cuda()
        targets = torch.randint(0, 2, (2, 16, 16)).cuda()
        
        # Update should work and move tensors to CPU internally
        metrics.update(preds, targets)
        
        # Metrics should still be on CPU device
        assert metrics.device == torch.device('cpu'), "Metrics should be on CPU"
    
    def test_edge_case_single_class_prediction(self, device):
        """Test edge case where all predictions are the same class."""
        metrics = SegmentationMetrics(num_classes=2, compute_on_cpu=True)
        
        # All predictions are class 0
        preds = torch.zeros(2, 16, 16, dtype=torch.long).to(device)
        # Targets have mixed classes
        targets = torch.randint(0, 2, (2, 16, 16)).to(device)
        
        metrics.update(preds, targets)
        result = metrics.compute()
        
        # Should handle this case without errors
        assert all(isinstance(v, float) for v in result.values()), \
            "All metrics should be valid floats"


class TestDiceCoefficient:
    """Test cases for dice_coefficient function."""
    
    def test_dice_perfect_overlap(self, device):
        """Test Dice coefficient with perfect overlap."""
        batch_size, num_classes, H, W = 2, 2, 16, 16
        
        # Create deterministic identical predictions and targets
        # Use ones for predictable behavior
        preds = torch.ones(batch_size, num_classes, H, W).to(device)
        targets = preds.clone()
        
        dice = dice_coefficient(preds, targets)
        
        # Should be very close to 1.0 (with small smoothing factor, not exactly 1.0)
        assert dice > 0.99, f"Dice should be very close to 1.0 for identical predictions and targets, got {dice}"
    
    def test_dice_no_overlap(self, device):
        """Test Dice coefficient with no overlap."""
        batch_size, num_classes, H, W = 2, 2, 16, 16
        
        # Create predictions and targets with no overlap
        preds = torch.zeros(batch_size, num_classes, H, W).to(device)
        preds[:, 0] = 1.0  # All class 0
        
        targets = torch.zeros(batch_size, num_classes, H, W).to(device)
        targets[:, 1] = 1.0  # All class 1
        
        dice = dice_coefficient(preds, targets)
        
        # Dice should be 0 for no overlap (with smoothing, it will be close to 0)
        assert dice < 0.1, "Dice should be very low for no overlap"
    
    def test_dice_partial_overlap(self, device):
        """Test Dice coefficient with partial overlap."""
        batch_size, num_classes, H, W = 2, 2, 16, 16
        
        preds = torch.rand(batch_size, num_classes, H, W).to(device)
        targets = torch.rand(batch_size, num_classes, H, W).to(device)
        
        dice = dice_coefficient(preds, targets)
        
        assert 0 <= dice <= 1, "Dice should be in [0, 1]"
        assert isinstance(dice, torch.Tensor), "Should return a tensor"
        assert dice.dim() == 0, "Should return a scalar"
    
    def test_dice_with_smoothing(self, device):
        """Test that smoothing prevents division by zero."""
        batch_size, num_classes, H, W = 2, 2, 16, 16
        
        # All zeros - could cause division by zero without smoothing
        preds = torch.zeros(batch_size, num_classes, H, W).to(device)
        targets = torch.zeros(batch_size, num_classes, H, W).to(device)
        
        dice = dice_coefficient(preds, targets, smooth=1e-6)
        
        assert not torch.isnan(dice), "Smoothing should prevent NaN"
        assert not torch.isinf(dice), "Smoothing should prevent inf"
    
    def test_dice_different_smoothing_values(self, device):
        """Test Dice with different smoothing values."""
        batch_size, num_classes, H, W = 2, 2, 16, 16
        preds = torch.rand(batch_size, num_classes, H, W).to(device)
        targets = torch.rand(batch_size, num_classes, H, W).to(device)
        
        smoothing_values = [1e-8, 1e-6, 1e-4, 1e-2]
        
        for smooth in smoothing_values:
            dice = dice_coefficient(preds, targets, smooth=smooth)
            
            assert 0 <= dice <= 1, f"Dice should be in [0, 1] for smooth={smooth}"
            assert not torch.isnan(dice), f"Dice should not be NaN for smooth={smooth}"
    
    def test_dice_binary_segmentation(self, device):
        """Test Dice for binary segmentation."""
        batch_size, H, W = 2, 16, 16
        
        # Binary predictions (2 classes)
        preds = torch.zeros(batch_size, 2, H, W).to(device)
        preds[:, 0, :8, :] = 1.0  # Top half is class 0
        preds[:, 1, 8:, :] = 1.0  # Bottom half is class 1
        
        targets = torch.zeros(batch_size, 2, H, W).to(device)
        targets[:, 0, :8, :] = 1.0  # Same as preds
        targets[:, 1, 8:, :] = 1.0
        
        dice = dice_coefficient(preds, targets)
        
        # Should be very high (close to 1.0) for identical masks
        assert dice > 0.99, "Dice should be near 1.0 for identical binary masks"
    
    def test_dice_multiclass_segmentation(self, device):
        """Test Dice for multiclass segmentation."""
        batch_size, num_classes, H, W = 2, 5, 16, 16
        
        preds = torch.rand(batch_size, num_classes, H, W).to(device)
        targets = torch.rand(batch_size, num_classes, H, W).to(device)
        
        dice = dice_coefficient(preds, targets)
        
        assert 0 <= dice <= 1, "Dice should be in [0, 1] for multiclass"
        assert not torch.isnan(dice), "Dice should not be NaN"
    
    def test_dice_shape_handling(self, device):
        """Test that Dice handles different input shapes correctly."""
        # Small image
        preds_small = torch.rand(1, 2, 8, 8).to(device)
        targets_small = torch.rand(1, 2, 8, 8).to(device)
        dice_small = dice_coefficient(preds_small, targets_small)
        
        # Large image
        preds_large = torch.rand(1, 2, 64, 64).to(device)
        targets_large = torch.rand(1, 2, 64, 64).to(device)
        dice_large = dice_coefficient(preds_large, targets_large)
        
        assert 0 <= dice_small <= 1, "Should handle small images"
        assert 0 <= dice_large <= 1, "Should handle large images"


class TestMetricsIntegration:
    """Integration tests for metrics."""
    
    def test_metrics_in_training_loop_simulation(self, device):
        """Simulate using metrics in a training loop."""
        metrics = SegmentationMetrics(num_classes=2, compute_on_cpu=True)
        
        # Simulate training epochs
        for epoch in range(3):
            metrics.reset()
            
            # Simulate batches
            for batch in range(5):
                preds = torch.randint(0, 2, (2, 16, 16)).to(device)
                targets = torch.randint(0, 2, (2, 16, 16)).to(device)
                metrics.update(preds, targets)
            
            # Compute epoch metrics
            result = metrics.compute()
            
            assert all(0 <= v <= 1 for v in result.values()), \
                f"Epoch {epoch}: All metrics should be in [0, 1]"
    
    def test_metrics_with_model_output(self, device):
        """Test metrics with actual model-like outputs."""
        metrics = SegmentationMetrics(num_classes=2, compute_on_cpu=True)
        
        # Simulate model output (logits after softmax)
        batch_size, num_classes, H, W = 4, 2, 32, 32
        logits = torch.randn(batch_size, num_classes, H, W).to(device)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Ground truth
        targets = torch.randint(0, num_classes, (batch_size, H, W)).to(device)
        
        # Update metrics with probabilities
        metrics.update(probs, targets)
        result = metrics.compute()
        
        assert all(isinstance(v, float) for v in result.values()), \
            "Should handle model-like outputs"
    
    def test_metrics_batch_consistency(self, device):
        """Test that metrics are consistent with batch size."""
        num_classes = 2
        H, W = 16, 16
        
        # Create consistent data
        torch.manual_seed(42)
        preds_large = torch.randint(0, num_classes, (10, H, W)).to(device)
        targets_large = torch.randint(0, num_classes, (10, H, W)).to(device)
        
        # Compute metrics on full batch
        metrics1 = SegmentationMetrics(num_classes=num_classes, compute_on_cpu=True)
        metrics1.update(preds_large, targets_large)
        result1 = metrics1.compute()
        
        # Compute metrics on split batches
        metrics2 = SegmentationMetrics(num_classes=num_classes, compute_on_cpu=True)
        metrics2.update(preds_large[:5], targets_large[:5])
        metrics2.update(preds_large[5:], targets_large[5:])
        result2 = metrics2.compute()
        
        # Results should be very close (might have small numerical differences)
        for key in result1.keys():
            assert abs(result1[key] - result2[key]) < 0.01, \
                f"{key} should be similar for full batch vs split batches"
    
    @pytest.mark.gpu
    def test_metrics_gpu_memory_efficiency(self):
        """Test that compute_on_cpu actually saves GPU memory."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create metrics with CPU computation
        metrics = SegmentationMetrics(num_classes=2, device=device, compute_on_cpu=True)
        
        # Add large batch
        preds = torch.randint(0, 2, (16, 512, 512)).cuda()
        targets = torch.randint(0, 2, (16, 512, 512)).cuda()
        
        mem_before = torch.cuda.memory_allocated()
        
        metrics.update(preds, targets)
        
        mem_after = torch.cuda.memory_allocated()
        
        # The preds and targets are moved to CPU inside update, so GPU memory
        # should not increase significantly (metrics are stored on CPU)
        # Allow for some increase due to temporary operations
        mem_increase = mem_after - mem_before
        
        # Memory increase should be reasonable (not keeping full copies on GPU)
        # This is a soft check since exact memory behavior can vary
        assert mem_increase < preds.element_size() * preds.nelement(), \
            "compute_on_cpu should reduce GPU memory usage for metric storage"
