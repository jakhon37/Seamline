"""
Unit tests for segmentation models.
"""
import pytest
import torch
import torch.nn as nn

from src.models.segmentation import create_model, get_model_info


class TestCreateModel:
    """Test cases for create_model function."""
    
    def test_create_unet(self):
        """Test creating a U-Net model."""
        model = create_model(
            architecture="Unet",
            encoder="resnet18",
            encoder_weights=None,
            num_classes=2,
            activation="softmax2d"
        )
        
        assert model is not None, "Model should be created"
        assert hasattr(model, 'forward'), "Model should have forward method"
    
    def test_create_unetplusplus(self):
        """Test creating a U-Net++ model."""
        model = create_model(
            architecture="UnetPlusPlus",
            encoder="resnet18",
            encoder_weights=None,
            num_classes=3
        )
        
        assert model is not None, "Model should be created"
    
    def test_create_deeplabv3(self):
        """Test creating a DeepLabV3 model."""
        model = create_model(
            architecture="DeepLabV3",
            encoder="resnet18",
            encoder_weights=None,
            num_classes=2
        )
        
        assert model is not None, "Model should be created"
    
    def test_create_deeplabv3plus(self):
        """Test creating a DeepLabV3+ model."""
        model = create_model(
            architecture="DeepLabV3Plus",
            encoder="resnet18",
            encoder_weights=None,
            num_classes=2
        )
        
        assert model is not None, "Model should be created"
    
    def test_create_fpn(self):
        """Test creating an FPN model."""
        model = create_model(
            architecture="FPN",
            encoder="resnet18",
            encoder_weights=None,
            num_classes=2
        )
        
        assert model is not None, "Model should be created"
    
    def test_create_pspnet(self):
        """Test creating a PSPNet model."""
        model = create_model(
            architecture="PSPNet",
            encoder="resnet18",
            encoder_weights=None,
            num_classes=2
        )
        
        assert model is not None, "Model should be created"
    
    def test_invalid_architecture_raises_error(self):
        """Test that invalid architecture raises ValueError."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            create_model(
                architecture="InvalidArch",
                encoder="resnet18",
                num_classes=2
            )
    
    def test_different_encoders(self):
        """Test creating models with different encoders."""
        encoders = ["resnet18", "resnet34", "resnet50"]
        
        for encoder in encoders:
            model = create_model(
                architecture="Unet",
                encoder=encoder,
                encoder_weights=None,
                num_classes=2
            )
            
            assert model is not None, f"Model with {encoder} should be created"
    
    def test_different_num_classes(self):
        """Test creating models with different number of classes."""
        for num_classes in [2, 3, 5, 10]:
            model = create_model(
                architecture="Unet",
                encoder="resnet18",
                encoder_weights=None,
                num_classes=num_classes
            )
            
            assert model is not None, f"Model with {num_classes} classes should be created"
    
    def test_activation_options(self):
        """Test different activation functions."""
        activations = ["softmax2d", "sigmoid", None]
        
        for activation in activations:
            model = create_model(
                architecture="Unet",
                encoder="resnet18",
                encoder_weights=None,
                num_classes=2,
                activation=activation
            )
            
            assert model is not None, f"Model with {activation} activation should be created"
    
    def test_gradient_checkpointing(self):
        """Test gradient checkpointing flag (should not raise errors)."""
        # Test with gradient checkpointing enabled
        model = create_model(
            architecture="Unet",
            encoder="resnet18",
            encoder_weights=None,
            num_classes=2,
            gradient_checkpointing=True
        )
        
        assert model is not None, "Model with gradient checkpointing should be created"
    
    def test_model_forward_pass(self, device):
        """Test model forward pass with dummy input."""
        model = create_model(
            architecture="Unet",
            encoder="resnet18",
            encoder_weights=None,
            num_classes=2,
            activation="softmax2d"
        )
        
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (batch_size, 2, 224, 224), \
            f"Output shape should be (2, 2, 224, 224), got {output.shape}"
        
        # Check softmax output
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size, 224, 224).to(device), atol=1e-5), \
            "Softmax output should sum to 1 along class dimension"
    
    def test_model_forward_different_sizes(self, device):
        """Test model with different input sizes."""
        model = create_model(
            architecture="Unet",
            encoder="resnet18",
            encoder_weights=None,
            num_classes=2
        )
        
        model = model.to(device)
        model.eval()
        
        # Test different input sizes
        sizes = [(128, 128), (256, 256), (512, 512)]
        
        for h, w in sizes:
            input_tensor = torch.randn(1, 3, h, w).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            assert output.shape == (1, 2, h, w), \
                f"Output shape should match input size for {h}x{w}"
    
    def test_model_trainable(self, device):
        """Test that model parameters are trainable."""
        model = create_model(
            architecture="Unet",
            encoder="resnet18",
            encoder_weights=None,
            num_classes=2
        )
        
        model = model.to(device)
        model.train()
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert trainable_params > 0, "Model should have trainable parameters"
        
        # Test backward pass
        input_tensor = torch.randn(1, 3, 224, 224).to(device)
        target = torch.randint(0, 2, (1, 224, 224)).to(device)
        
        output = model(input_tensor)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # Check that gradients are computed
        has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_gradients, "Model should have gradients after backward pass"


class TestGetModelInfo:
    """Test cases for get_model_info function."""
    
    def test_get_model_info(self):
        """Test getting model information."""
        model = create_model(
            architecture="Unet",
            encoder="resnet18",
            encoder_weights=None,
            num_classes=2
        )
        
        info = get_model_info(model)
        
        assert isinstance(info, dict), "Should return a dictionary"
        assert "total_params" in info, "Should contain total_params"
        assert "trainable_params" in info, "Should contain trainable_params"
        assert "total_params_m" in info, "Should contain total_params_m"
        assert "trainable_params_m" in info, "Should contain trainable_params_m"
    
    def test_model_info_values(self):
        """Test that model info values are correct."""
        model = create_model(
            architecture="Unet",
            encoder="resnet18",
            encoder_weights=None,
            num_classes=2
        )
        
        info = get_model_info(model)
        
        # Check that values are positive
        assert info["total_params"] > 0, "Total params should be positive"
        assert info["trainable_params"] > 0, "Trainable params should be positive"
        
        # Check that trainable <= total
        assert info["trainable_params"] <= info["total_params"], \
            "Trainable params should be <= total params"
        
        # Check conversion to millions
        assert abs(info["total_params_m"] - info["total_params"] / 1e6) < 1e-5, \
            "total_params_m should be total_params / 1e6"
    
    def test_model_info_frozen_params(self):
        """Test model info with frozen parameters."""
        model = create_model(
            architecture="Unet",
            encoder="resnet18",
            encoder_weights=None,
            num_classes=2
        )
        
        # Freeze encoder parameters
        for param in model.encoder.parameters():
            param.requires_grad = False
        
        info = get_model_info(model)
        
        # Trainable should be less than total
        assert info["trainable_params"] < info["total_params"], \
            "Trainable params should be less than total when some are frozen"
    
    def test_model_info_different_models(self):
        """Test model info for different architectures."""
        architectures = ["Unet", "FPN", "PSPNet"]
        
        for arch in architectures:
            model = create_model(
                architecture=arch,
                encoder="resnet18",
                encoder_weights=None,
                num_classes=2
            )
            
            info = get_model_info(model)
            
            assert info["total_params"] > 0, f"{arch} should have parameters"


class TestModelIntegration:
    """Integration tests for models."""
    
    def test_model_batch_processing(self, device):
        """Test model with batched inputs."""
        model = create_model(
            architecture="Unet",
            encoder="resnet18",
            encoder_weights=None,
            num_classes=2
        )
        
        model = model.to(device)
        model.eval()
        
        batch_sizes = [1, 4, 8]
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            assert output.shape[0] == batch_size, \
                f"Output batch size should match input for batch_size={batch_size}"
    
    def test_model_memory_efficient(self, device):
        """Test that model runs without OOM on reasonable input."""
        model = create_model(
            architecture="Unet",
            encoder="resnet18",
            encoder_weights=None,
            num_classes=2
        )
        
        model = model.to(device)
        model.eval()
        
        # Test with reasonably sized batch
        input_tensor = torch.randn(2, 3, 256, 256).to(device)
        
        try:
            with torch.no_grad():
                output = model(input_tensor)
            success = True
        except RuntimeError as e:
            if "out of memory" in str(e):
                success = False
            else:
                raise
        
        assert success, "Model should run without OOM on reasonable input"
    
    @pytest.mark.gpu
    def test_model_cuda_compatibility(self):
        """Test model CUDA compatibility."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = create_model(
            architecture="Unet",
            encoder="resnet18",
            encoder_weights=None,
            num_classes=2
        )
        
        # Move to CUDA
        model = model.cuda()
        
        # Test forward pass on CUDA
        input_tensor = torch.randn(1, 3, 224, 224).cuda()
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.is_cuda, "Output should be on CUDA"
        assert output.shape == (1, 2, 224, 224), "Output shape should be correct"
