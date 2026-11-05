"""
Unit tests for loss functions.
"""
import pytest
import torch
import torch.nn as nn

from src.losses.combined_loss import CombinedLoss, DiceLoss
from src.losses.focal_loss import FocalLoss
from src.losses.loss_factory import create_loss


class TestFocalLoss:
    """Test cases for FocalLoss."""

    def test_focal_loss_initialization(self):
        """Test FocalLoss initialization."""
        loss_fn = FocalLoss(gamma=2.0)

        assert loss_fn.gamma == 2.0, "Gamma should be set correctly"
        assert loss_fn.reduction == "mean", "Default reduction should be mean"
        assert loss_fn.__name__ == "focal_loss", "Should have correct name"

    def test_focal_loss_with_weights(self):
        """Test FocalLoss with class weights."""
        weights = torch.FloatTensor([1.0, 2.0])
        loss_fn = FocalLoss(gamma=2.0, weight=weights)

        assert loss_fn._weight_values == [1.0, 2.0], "Weights should be stored as list"

    def test_focal_loss_forward_class_indices(self, device):
        """Test FocalLoss forward pass with class indices."""
        loss_fn = FocalLoss(gamma=2.0)

        # Create dummy data
        batch_size, num_classes, H, W = 2, 2, 16, 16
        logits = torch.randn(batch_size, num_classes, H, W).to(device)
        targets = torch.randint(0, num_classes, (batch_size, H, W)).to(device)

        loss = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor), "Should return a tensor"
        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be inf"

    def test_focal_loss_forward_onehot(self, device):
        """Test FocalLoss forward pass with one-hot encoded targets."""
        loss_fn = FocalLoss(gamma=2.0)

        batch_size, num_classes, H, W = 2, 2, 16, 16
        logits = torch.randn(batch_size, num_classes, H, W).to(device)

        # Create one-hot encoded targets
        targets_indices = torch.randint(0, num_classes, (batch_size, H, W))
        targets_onehot = torch.zeros(batch_size, num_classes, H, W)
        targets_onehot.scatter_(1, targets_indices.unsqueeze(1), 1)
        targets_onehot = targets_onehot.to(device)

        loss = loss_fn(logits, targets_onehot)

        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_focal_loss_different_gammas(self, device):
        """Test FocalLoss with different gamma values."""
        batch_size, num_classes, H, W = 2, 2, 16, 16
        logits = torch.randn(batch_size, num_classes, H, W).to(device)
        targets = torch.randint(0, num_classes, (batch_size, H, W)).to(device)

        gammas = [0.0, 1.0, 2.0, 3.0]
        losses = []

        for gamma in gammas:
            loss_fn = FocalLoss(gamma=gamma)
            loss = loss_fn(logits, targets)
            losses.append(loss.item())

        # All losses should be valid
        for loss in losses:
            assert loss >= 0, "All losses should be non-negative"

    def test_focal_loss_backward(self, device):
        """Test that FocalLoss supports backpropagation."""
        loss_fn = FocalLoss(gamma=2.0)

        logits = torch.randn(2, 2, 16, 16, device=device, requires_grad=True)
        targets = torch.randint(0, 2, (2, 16, 16), device=device)

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None, "Should compute gradients"
        assert not torch.isnan(logits.grad).any(), "Gradients should not be NaN"

    def test_focal_loss_reduction_modes(self, device):
        """Test different reduction modes."""
        logits = torch.randn(2, 2, 16, 16).to(device)
        targets = torch.randint(0, 2, (2, 16, 16)).to(device)

        # Test mean reduction
        loss_mean = FocalLoss(gamma=2.0, reduction="mean")(logits, targets)
        assert loss_mean.dim() == 0, "Mean reduction should return scalar"

        # Test sum reduction
        loss_sum = FocalLoss(gamma=2.0, reduction="sum")(logits, targets)
        assert loss_sum.dim() == 0, "Sum reduction should return scalar"
        assert loss_sum > loss_mean, "Sum should be larger than mean"

        # Test none reduction
        loss_none = FocalLoss(gamma=2.0, reduction="none")(logits, targets)
        assert loss_none.shape == targets.shape, "None reduction should preserve shape"


class TestDiceLoss:
    """Test cases for DiceLoss."""

    def test_dice_loss_initialization(self):
        """Test DiceLoss initialization."""
        loss_fn = DiceLoss(mode="multiclass", from_logits=True)

        assert loss_fn.__name__ == "dice_loss", "Should have correct name"

    def test_dice_loss_forward_class_indices(self, device):
        """Test DiceLoss forward pass with class indices."""
        loss_fn = DiceLoss(mode="multiclass", from_logits=True)

        batch_size, num_classes, H, W = 2, 2, 16, 16
        logits = torch.randn(batch_size, num_classes, H, W).to(device)
        targets = torch.randint(0, num_classes, (batch_size, H, W)).to(device)

        loss = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor), "Should return a tensor"
        assert loss.dim() == 0, "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_dice_loss_forward_onehot(self, device):
        """Test DiceLoss forward pass with one-hot encoded targets."""
        loss_fn = DiceLoss(mode="multiclass", from_logits=True)

        batch_size, num_classes, H, W = 2, 2, 16, 16
        logits = torch.randn(batch_size, num_classes, H, W).to(device)

        # Create one-hot encoded targets
        targets_indices = torch.randint(0, num_classes, (batch_size, H, W))
        targets_onehot = torch.zeros(batch_size, num_classes, H, W)
        targets_onehot.scatter_(1, targets_indices.unsqueeze(1), 1)
        targets_onehot = targets_onehot.to(device)

        loss = loss_fn(logits, targets_onehot)

        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_dice_loss_backward(self, device):
        """Test that DiceLoss supports backpropagation."""
        loss_fn = DiceLoss(mode="multiclass", from_logits=True)

        logits = torch.randn(2, 2, 16, 16, device=device, requires_grad=True)
        targets = torch.randint(0, 2, (2, 16, 16), device=device)

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None, "Should compute gradients"


class TestCombinedLoss:
    """Test cases for CombinedLoss."""

    def test_combined_loss_initialization(self):
        """Test CombinedLoss initialization."""
        focal = FocalLoss(gamma=2.0)
        dice = DiceLoss(mode="multiclass", from_logits=True)

        combined = CombinedLoss(focal, dice, alpha=0.5)

        assert combined.alpha == 0.5, "Alpha should be set correctly"
        assert combined.primary_loss is focal, "Primary loss should be stored"
        assert combined.secondary_loss is dice, "Secondary loss should be stored"

    def test_combined_loss_forward(self, device):
        """Test CombinedLoss forward pass."""
        focal = FocalLoss(gamma=2.0)
        dice = DiceLoss(mode="multiclass", from_logits=True)
        combined = CombinedLoss(focal, dice, alpha=0.5)

        batch_size, num_classes, H, W = 2, 2, 16, 16
        logits = torch.randn(batch_size, num_classes, H, W).to(device)
        targets = torch.randint(0, num_classes, (batch_size, H, W)).to(device)

        loss_combined = combined(logits, targets)
        loss_focal = focal(logits, targets)
        loss_dice = dice(logits, targets)

        # Check that combined loss is weighted average
        expected = 0.5 * loss_focal + 0.5 * loss_dice
        assert torch.allclose(
            loss_combined, expected, atol=1e-5
        ), "Combined loss should be weighted average"

    def test_combined_loss_different_alphas(self, device):
        """Test CombinedLoss with different alpha values."""
        focal = FocalLoss(gamma=2.0)
        dice = DiceLoss(mode="multiclass", from_logits=True)

        logits = torch.randn(2, 2, 16, 16).to(device)
        targets = torch.randint(0, 2, (2, 16, 16)).to(device)

        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

        for alpha in alphas:
            combined = CombinedLoss(focal, dice, alpha=alpha)
            loss = combined(logits, targets)

            assert not torch.isnan(loss), f"Loss should not be NaN for alpha={alpha}"
            assert loss.item() >= 0, f"Loss should be non-negative for alpha={alpha}"

    def test_combined_loss_backward(self, device):
        """Test that CombinedLoss supports backpropagation."""
        focal = FocalLoss(gamma=2.0)
        dice = DiceLoss(mode="multiclass", from_logits=True)
        combined = CombinedLoss(focal, dice, alpha=0.5)

        logits = torch.randn(2, 2, 16, 16, device=device, requires_grad=True)
        targets = torch.randint(0, 2, (2, 16, 16), device=device)

        loss = combined(logits, targets)
        loss.backward()

        assert logits.grad is not None, "Should compute gradients"


class TestCreateLoss:
    """Test cases for create_loss factory function."""

    def test_create_focal_loss(self, sample_config):
        """Test creating focal loss."""
        sample_config["loss"] = {"type": "focal", "focal_gamma": 2.0}

        loss_fn = create_loss(sample_config)

        assert isinstance(loss_fn, FocalLoss), "Should create FocalLoss"
        assert loss_fn.gamma == 2.0, "Gamma should be set correctly"

    def test_create_dice_loss(self, sample_config):
        """Test creating dice loss."""
        sample_config["loss"] = {"type": "dice"}

        loss_fn = create_loss(sample_config)

        assert isinstance(loss_fn, DiceLoss), "Should create DiceLoss"

    def test_create_cross_entropy_loss(self, sample_config):
        """Test creating cross entropy loss."""
        sample_config["loss"] = {"type": "cross_entropy"}

        loss_fn = create_loss(sample_config)

        assert isinstance(loss_fn, nn.CrossEntropyLoss), "Should create CrossEntropyLoss"
        assert loss_fn.__name__ == "cross_entropy_loss", "Should have correct name"

    def test_create_combined_loss(self, sample_config):
        """Test creating combined loss."""
        sample_config["loss"] = {"type": "combined", "focal_gamma": 2.0, "combined_alpha": 0.6}

        loss_fn = create_loss(sample_config)

        assert isinstance(loss_fn, CombinedLoss), "Should create CombinedLoss"
        assert loss_fn.alpha == 0.6, "Alpha should be set correctly"

    def test_create_tversky_loss(self, sample_config):
        """Test creating tversky loss."""
        sample_config["loss"] = {"type": "tversky"}

        loss_fn = create_loss(sample_config)

        assert loss_fn is not None, "Should create TverskyLoss"
        assert loss_fn.__name__ == "tversky_loss", "Should have correct name"

    def test_create_loss_with_class_weights(self, sample_config):
        """Test creating loss with class weights."""
        sample_config["loss"] = {"type": "cross_entropy", "class_weights": [1.0, 2.0]}

        loss_fn = create_loss(sample_config)

        assert isinstance(loss_fn, nn.CrossEntropyLoss), "Should create loss"
        # Weights are stored internally and applied during forward pass

    def test_create_focal_loss_with_weights(self, sample_config):
        """Test creating focal loss with class weights."""
        sample_config["loss"] = {"type": "focal", "focal_gamma": 2.0, "class_weights": [1.0, 3.0]}

        loss_fn = create_loss(sample_config)

        assert isinstance(loss_fn, FocalLoss), "Should create FocalLoss"
        assert loss_fn._weight_values == [1.0, 3.0], "Weights should be set"

    def test_invalid_loss_type_raises_error(self, sample_config):
        """Test that invalid loss type raises ValueError."""
        sample_config["loss"] = {"type": "invalid_loss"}

        with pytest.raises(ValueError, match="Unknown loss type"):
            create_loss(sample_config)

    def test_loss_forward_compatibility(self, sample_config, device):
        """Test that all loss types can process inputs."""
        loss_types = ["focal", "dice", "cross_entropy", "combined", "tversky"]

        batch_size, num_classes, H, W = 2, 2, 16, 16
        logits = torch.randn(batch_size, num_classes, H, W).to(device)
        targets = torch.randint(0, num_classes, (batch_size, H, W)).to(device)

        for loss_type in loss_types:
            sample_config["loss"] = {"type": loss_type}
            if loss_type == "focal":
                sample_config["loss"]["focal_gamma"] = 2.0
            if loss_type == "combined":
                sample_config["loss"]["focal_gamma"] = 2.0
                sample_config["loss"]["combined_alpha"] = 0.5

            loss_fn = create_loss(sample_config, device)
            loss = loss_fn(logits, targets)

            assert not torch.isnan(loss), f"{loss_type} should not produce NaN"
            assert not torch.isinf(loss), f"{loss_type} should not produce inf"

    def test_loss_with_onehot_targets(self, sample_config, device):
        """Test that losses work with one-hot encoded targets."""
        loss_types = ["focal", "dice", "combined"]

        batch_size, num_classes, H, W = 2, 2, 16, 16
        logits = torch.randn(batch_size, num_classes, H, W).to(device)

        # Create one-hot targets
        targets_indices = torch.randint(0, num_classes, (batch_size, H, W))
        targets_onehot = torch.zeros(batch_size, num_classes, H, W)
        targets_onehot.scatter_(1, targets_indices.unsqueeze(1), 1)
        targets_onehot = targets_onehot.to(device)

        for loss_type in loss_types:
            sample_config["loss"] = {"type": loss_type}
            if loss_type in ["focal", "combined"]:
                sample_config["loss"]["focal_gamma"] = 2.0
            if loss_type == "combined":
                sample_config["loss"]["combined_alpha"] = 0.5

            loss_fn = create_loss(sample_config, device)
            loss = loss_fn(logits, targets_onehot)

            assert not torch.isnan(loss), f"{loss_type} should handle one-hot targets"

    def test_loss_multiclass(self, sample_config, device):
        """Test losses with more than 2 classes."""
        sample_config["loss"] = {"type": "cross_entropy"}
        loss_fn = create_loss(sample_config, device)

        num_classes = 5
        logits = torch.randn(2, num_classes, 16, 16).to(device)
        targets = torch.randint(0, num_classes, (2, 16, 16)).to(device)

        loss = loss_fn(logits, targets)

        assert not torch.isnan(loss), "Should handle multiclass segmentation"
        assert loss.item() >= 0, "Loss should be non-negative"


class TestLossEdgeCases:
    """Test edge cases for loss functions."""

    def test_loss_with_perfect_prediction(self, device):
        """Test loss when predictions are perfect."""
        loss_fn = nn.CrossEntropyLoss()

        # Create perfect predictions
        batch_size, num_classes, H, W = 2, 2, 16, 16
        targets = torch.randint(0, num_classes, (batch_size, H, W)).to(device)

        # Create logits that perfectly match targets
        logits = torch.zeros(batch_size, num_classes, H, W).to(device)
        for b in range(batch_size):
            for h in range(H):
                for w in range(W):
                    logits[b, targets[b, h, w], h, w] = 10.0  # High confidence

        loss = loss_fn(logits, targets)

        # Loss should be very small (near zero)
        assert loss.item() < 0.1, "Loss should be near zero for perfect predictions"

    def test_loss_with_uniform_prediction(self, device):
        """Test loss with uniform (uncertain) predictions."""
        loss_fn = nn.CrossEntropyLoss()

        # Create uniform predictions (equal logits for all classes)
        batch_size, num_classes, H, W = 2, 2, 16, 16
        logits = torch.zeros(batch_size, num_classes, H, W).to(device)
        targets = torch.zeros(batch_size, H, W, dtype=torch.long).to(device)

        loss = loss_fn(logits, targets)

        # Loss should be -log(1/num_classes)
        expected_loss = -torch.log(torch.tensor(1.0 / num_classes))
        assert (
            abs(loss.item() - expected_loss.item()) < 0.01
        ), "Loss should match expected value for uniform predictions"

    def test_loss_with_single_pixel(self, device):
        """Test loss with minimal input size."""
        loss_fn = FocalLoss(gamma=2.0)

        # Single pixel prediction
        logits = torch.randn(1, 2, 1, 1).to(device)
        targets = torch.randint(0, 2, (1, 1, 1)).to(device)

        loss = loss_fn(logits, targets)

        assert not torch.isnan(loss), "Should handle single pixel input"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_loss_gradient_flow(self, device):
        """Test that gradients flow through all loss types."""
        loss_types = [
            FocalLoss(gamma=2.0),
            DiceLoss(mode="multiclass", from_logits=True),
            nn.CrossEntropyLoss(),
        ]

        for loss_fn in loss_types:
            logits = torch.randn(2, 2, 16, 16, device=device, requires_grad=True)
            targets = torch.randint(0, 2, (2, 16, 16), device=device)

            loss = loss_fn(logits, targets)
            loss.backward()

            assert logits.grad is not None, f"{loss_fn.__class__.__name__} should produce gradients"
            assert not torch.isnan(
                logits.grad
            ).any(), f"{loss_fn.__class__.__name__} gradients should not be NaN"
