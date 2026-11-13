"""
Unit tests for SegmentationDataset class.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.datamodule import create_dataloaders
from src.data.dataset import SegmentationDataset, get_image_mask_pairs
from src.data.transforms import get_val_transforms


class TestSegmentationDataset:
    """Test cases for SegmentationDataset class."""

    def test_initialization(self, tmp_path):
        """Test dataset initialization with valid paths."""
        # Create dummy files
        img_paths = []
        mask_paths = []
        for i in range(5):
            img_path = tmp_path / f"img_{i}.png"
            mask_path = tmp_path / f"mask_{i}.png"

            Image.new("RGB", (100, 100)).save(img_path)
            Image.new("L", (100, 100)).save(mask_path)

            img_paths.append(str(img_path))
            mask_paths.append(str(mask_path))

        dataset = SegmentationDataset(image_paths=img_paths, mask_paths=mask_paths, num_classes=2)

        assert len(dataset) == 5, "Dataset length should match number of samples"
        assert dataset.num_classes == 2, "Number of classes should be 2"
        assert not dataset.use_onehot, "Default should be class indices mode"
        assert dataset.cache is None, "Cache should be None when caching disabled"

    def test_initialization_with_invalid_extensions(self, tmp_path):
        """Test that invalid file extensions are filtered out."""
        img_paths = [
            str(tmp_path / "img_1.png"),
            str(tmp_path / "img_2.txt"),  # Invalid
            str(tmp_path / "img_3.jpg"),
        ]
        mask_paths = [
            str(tmp_path / "mask_1.png"),
            str(tmp_path / "mask_2.txt"),  # Invalid - should be filtered too
            str(tmp_path / "mask_3.jpg"),
        ]

        # Create actual files
        for p in [img_paths[0], img_paths[2]]:
            Image.new("RGB", (100, 100)).save(p)
        for p in [mask_paths[0], mask_paths[2]]:
            Image.new("L", (100, 100)).save(p)

        dataset = SegmentationDataset(image_paths=img_paths, mask_paths=mask_paths, num_classes=2)

        assert len(dataset) == 2, "Should filter out invalid extensions"

    def test_mismatched_lengths_raises_error(self, tmp_path):
        """Test that mismatched image and mask counts raise an error."""
        img_paths = [str(tmp_path / "img_1.png")]
        mask_paths = [str(tmp_path / "mask_1.png"), str(tmp_path / "mask_2.png")]

        # Create files
        Image.new("RGB", (100, 100)).save(img_paths[0])
        for p in mask_paths:
            Image.new("L", (100, 100)).save(p)

        with pytest.raises(AssertionError, match="Number of images.*!=.*number of masks"):
            SegmentationDataset(image_paths=img_paths, mask_paths=mask_paths, num_classes=2)

    def test_getitem_class_indices_mode(self, tmp_path):
        """Test __getitem__ returns class indices (default mode)."""
        # Create test image and mask
        img_path = tmp_path / "img.png"
        mask_path = tmp_path / "mask.png"

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask_array = np.random.randint(0, 2, (100, 100), dtype=np.uint8)

        Image.fromarray(img_array).save(img_path)
        Image.fromarray(mask_array).save(mask_path)

        dataset = SegmentationDataset(
            image_paths=[str(img_path)],
            mask_paths=[str(mask_path)],
            num_classes=2,
            use_onehot=False,
        )

        image, mask = dataset[0]

        # Check shapes
        assert isinstance(image, np.ndarray), "Image should be numpy array"
        assert isinstance(mask, torch.Tensor), "Mask should be tensor"
        assert mask.dtype == torch.int64, "Mask should be int64 for class indices"
        assert mask.dim() == 2, "Mask should be 2D [H, W]"
        assert mask.max() < 2, "Mask values should be less than num_classes"
        assert mask.min() >= 0, "Mask values should be non-negative"

    def test_getitem_onehot_mode(self, tmp_path):
        """Test __getitem__ returns one-hot encoded masks (legacy mode)."""
        img_path = tmp_path / "img.png"
        mask_path = tmp_path / "mask.png"

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask_array = np.random.randint(0, 2, (100, 100), dtype=np.uint8)

        Image.fromarray(img_array).save(img_path)
        Image.fromarray(mask_array).save(mask_path)

        dataset = SegmentationDataset(
            image_paths=[str(img_path)], mask_paths=[str(mask_path)], num_classes=2, use_onehot=True
        )

        image, mask = dataset[0]

        # Check one-hot encoding
        assert isinstance(mask, torch.Tensor), "Mask should be tensor"
        assert mask.dtype == torch.float32, "One-hot mask should be float32"
        assert mask.dim() == 3, "One-hot mask should be 3D [C, H, W]"
        assert mask.shape[0] == 2, "First dimension should equal num_classes"
        assert torch.all((mask == 0) | (mask == 1)), "One-hot values should be 0 or 1"
        assert torch.allclose(
            mask.sum(dim=0), torch.ones_like(mask[0])
        ), "Each pixel should have exactly one class"

    def test_caching_functionality(self, tmp_path):
        """Test that caching stores and retrieves samples correctly."""
        img_path = tmp_path / "img.png"
        mask_path = tmp_path / "mask.png"

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask_array = np.random.randint(0, 2, (100, 100), dtype=np.uint8)

        Image.fromarray(img_array).save(img_path)
        Image.fromarray(mask_array).save(mask_path)

        dataset = SegmentationDataset(
            image_paths=[str(img_path)],
            mask_paths=[str(mask_path)],
            num_classes=2,
            cache_in_memory=True,
        )

        assert isinstance(dataset.cache, dict), "Cache should be initialized as dict"
        assert 0 not in dataset.cache, "Cache should be empty initially"

        # First access - should load and cache
        image1, mask1 = dataset[0]
        assert 0 in dataset.cache, "Sample should be cached after first access"

        # Second access - should retrieve from cache
        image2, mask2 = dataset[0]

        # Verify same objects are returned (from cache)
        assert image1 is image2, "Should return same image object from cache"
        assert mask1 is mask2, "Should return same mask object from cache"

    def test_mask_clipping(self, tmp_path):
        """Test that mask values are clipped to valid range."""
        img_path = tmp_path / "img.png"
        mask_path = tmp_path / "mask.png"

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Create mask with out-of-range values
        mask_array = np.random.randint(0, 5, (100, 100), dtype=np.uint8)

        Image.fromarray(img_array).save(img_path)
        Image.fromarray(mask_array).save(mask_path)

        dataset = SegmentationDataset(
            image_paths=[str(img_path)], mask_paths=[str(mask_path)], num_classes=2
        )

        _, mask = dataset[0]

        assert mask.max() <= 1, "Mask values should be clipped to num_classes-1"
        assert mask.min() >= 0, "Mask values should be clipped to 0"

    def test_transform_integration(self, tmp_path):
        """Test that transforms are applied correctly."""
        import albumentations as A

        img_path = tmp_path / "img.png"
        mask_path = tmp_path / "mask.png"

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask_array = np.random.randint(0, 2, (100, 100), dtype=np.uint8)

        Image.fromarray(img_array).save(img_path)
        Image.fromarray(mask_array).save(mask_path)

        # Create transform that resizes images (without ToTensorV2)
        transform = A.Compose(
            [
                A.Resize(64, 64),
            ]
        )

        dataset = SegmentationDataset(
            image_paths=[str(img_path)],
            mask_paths=[str(mask_path)],
            num_classes=2,
            transform=transform,
            use_onehot=False,
        )

        image, mask = dataset[0]

        # Check that transform was applied
        # Image is still numpy after albumentations (ToTensorV2 not used)
        assert isinstance(image, np.ndarray), "Image should be numpy array"
        assert image.shape[:2] == (64, 64), "Transform should resize image"
        # Mask is converted to tensor by dataset
        assert isinstance(mask, torch.Tensor), "Mask should be tensor"
        assert mask.shape == (64, 64), "Mask should also be resized"

    def test_empty_dataset(self):
        """Test behavior with empty dataset."""
        dataset = SegmentationDataset(image_paths=[], mask_paths=[], num_classes=2)

        assert len(dataset) == 0, "Empty dataset should have length 0"

    def test_multiclass_dataset(self, tmp_path):
        """Test dataset with more than 2 classes."""
        img_path = tmp_path / "img.png"
        mask_path = tmp_path / "mask.png"

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask_array = np.random.randint(0, 5, (100, 100), dtype=np.uint8)

        Image.fromarray(img_array).save(img_path)
        Image.fromarray(mask_array).save(mask_path)

        dataset = SegmentationDataset(
            image_paths=[str(img_path)], mask_paths=[str(mask_path)], num_classes=5, use_onehot=True
        )

        _, mask = dataset[0]

        assert mask.shape[0] == 5, "One-hot encoding should have 5 channels"
        assert mask.max() <= 1, "One-hot values should be 0 or 1"


class TestGetImageMaskPairs:
    """Test cases for get_image_mask_pairs function."""

    def test_basic_pairing(self, tmp_path):
        """Test basic image-mask pairing."""
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "masks"
        images_dir.mkdir()
        masks_dir.mkdir()

        # Create files
        for i in range(3):
            Image.new("RGB", (100, 100)).save(images_dir / f"sample_{i}.png")
            Image.new("L", (100, 100)).save(masks_dir / f"sample_{i}.png")

        img_paths, mask_paths = get_image_mask_pairs(str(images_dir), str(masks_dir))

        assert len(img_paths) == 3, "Should find all image files"
        assert len(mask_paths) == 3, "Should find all mask files"

        # Check pairing
        for img_path, mask_path in zip(img_paths, mask_paths):
            img_name = Path(img_path).stem
            mask_name = Path(mask_path).stem
            assert img_name == mask_name, f"Names should match: {img_name} vs {mask_name}"

    def test_sorting(self, tmp_path):
        """Test that files are sorted correctly."""
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "masks"
        images_dir.mkdir()
        masks_dir.mkdir()

        # Create files in non-alphabetical order
        for name in ["sample_10", "sample_2", "sample_1"]:
            Image.new("RGB", (100, 100)).save(images_dir / f"{name}.png")
            Image.new("L", (100, 100)).save(masks_dir / f"{name}.png")

        img_paths, mask_paths = get_image_mask_pairs(str(images_dir), str(masks_dir))

        # Extract names
        img_names = [Path(p).stem for p in img_paths]
        mask_names = [Path(p).stem for p in mask_paths]

        assert img_names == sorted(img_names), "Images should be sorted"
        assert mask_names == sorted(mask_names), "Masks should be sorted"

    def test_filter_mask_suffix(self, tmp_path):
        """Test that _mask.tif files are filtered from images."""
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "masks"
        images_dir.mkdir()
        masks_dir.mkdir()

        # Create normal images
        Image.new("RGB", (100, 100)).save(images_dir / "sample_1.png")
        # Create mask file in images dir (should be filtered)
        Image.new("L", (100, 100)).save(images_dir / "sample_1_mask.tif")

        # Create mask
        Image.new("L", (100, 100)).save(masks_dir / "sample_1.png")

        img_paths, mask_paths = get_image_mask_pairs(str(images_dir), str(masks_dir))

        assert len(img_paths) == 1, "Should only find regular image file"
        assert "_mask.tif" not in img_paths[0], "Should filter out _mask.tif files"

    def test_masks_named_with_a_mask_suffix(self, tmp_path):
        """a_mask.png is a.png's mask when no mask has the image's own name."""
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "masks"
        images_dir.mkdir()
        masks_dir.mkdir()
        for name in ("a", "b"):
            Image.new("RGB", (8, 8)).save(images_dir / f"{name}.png")
            Image.new("L", (8, 8)).save(masks_dir / f"{name}_mask.png")

        img_paths, mask_paths = get_image_mask_pairs(str(images_dir), str(masks_dir))

        assert [Path(p).name for p in img_paths] == ["a.png", "b.png"]
        assert [Path(p).name for p in mask_paths] == ["a_mask.png", "b_mask.png"]

    def test_a_mask_with_the_images_own_name_wins(self, tmp_path):
        """Names that differ only by _mask keep their own masks, as the app's tiles do: x/a.png
        and x_mask/a.png, or a.png and a_mask.png, in one project."""
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "masks"
        images_dir.mkdir()
        masks_dir.mkdir()
        names = ["p__0_0", "p_mask__0_0", "x%2Fa__0_0", "x_mask%2Fa__0_0"]
        for name in names:
            Image.new("RGB", (8, 8)).save(images_dir / f"{name}.png")
            Image.new("L", (8, 8)).save(masks_dir / f"{name}.png")
        # And an image without a mask of its own leaves another image's alone
        Image.new("RGB", (8, 8)).save(images_dir / "q.png")
        Image.new("RGB", (8, 8)).save(images_dir / "q_mask.png")
        Image.new("L", (8, 8)).save(masks_dir / "q_mask.png")

        with pytest.warns(UserWarning, match="Skipping 1 images without a mask, e.g. q.png"):
            img_paths, mask_paths = get_image_mask_pairs(str(images_dir), str(masks_dir))

        pairs = [(Path(i).name, Path(m).name) for i, m in zip(img_paths, mask_paths)]
        assert sorted(pairs) == sorted((f"{n}.png", f"{n}.png") for n in names + ["q_mask"])

    def test_multiple_extensions(self, tmp_path):
        """Test support for multiple file extensions."""
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "masks"
        images_dir.mkdir()
        masks_dir.mkdir()

        # Create files with different extensions
        Image.new("RGB", (100, 100)).save(images_dir / "img1.png")
        Image.new("RGB", (100, 100)).save(images_dir / "img2.jpg")
        Image.new("RGB", (100, 100)).save(images_dir / "img3.tif")

        Image.new("L", (100, 100)).save(masks_dir / "img1.png")
        Image.new("L", (100, 100)).save(masks_dir / "img2.png")
        Image.new("L", (100, 100)).save(masks_dir / "img3.png")

        img_paths, mask_paths = get_image_mask_pairs(str(images_dir), str(masks_dir))

        assert len(img_paths) == 3, "Should find images with various extensions"
        assert len(mask_paths) == 3, "Should find all masks"

    def test_empty_directory(self, tmp_path):
        """Test behavior with empty directories."""
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "masks"
        images_dir.mkdir()
        masks_dir.mkdir()

        img_paths, mask_paths = get_image_mask_pairs(str(images_dir), str(masks_dir))

        assert len(img_paths) == 0, "Should return empty list for empty directory"
        assert len(mask_paths) == 0, "Should return empty list for empty directory"


class TestWithTrainingPipeline:
    """Regression tests for the path that real training runs take."""

    def test_class_index_masks_with_tensor_transform(self, tmp_path):
        """Masks stay class indices when the transform already returns tensors."""
        img_path = tmp_path / "img.png"
        mask_path = tmp_path / "mask.png"
        Image.new("RGB", (40, 40)).save(img_path)
        Image.fromarray(np.ones((40, 40), dtype=np.uint8)).save(mask_path)

        dataset = SegmentationDataset(
            image_paths=[str(img_path)],
            mask_paths=[str(mask_path)],
            transform=get_val_transforms(height=32, width=32),
            num_classes=2,
        )
        image, mask = dataset[0]

        assert image.shape == (3, 32, 32)
        assert mask.shape == (32, 32)
        assert mask.dtype == torch.int64
        assert mask.max() == 1

    def test_create_dataloaders_with_test_folder(self, tmp_path):
        """A dataset with a test/ folder builds all three loaders."""
        for split in ("train", "val", "test"):
            for sub in ("images", "masks"):
                (tmp_path / "dataset" / split / sub).mkdir(parents=True)
            for i in range(2):
                Image.new("RGB", (40, 40)).save(
                    tmp_path / "dataset" / split / "images" / f"{i}.png"
                )
                Image.new("L", (40, 40)).save(tmp_path / "dataset" / split / "masks" / f"{i}.png")

        config = {
            "experiment": {"seed": 42},
            "data": {
                "base_dir": str(tmp_path),
                "dataset_dir": "dataset",
                "train_dir": "train",
                "val_dir": "val",
                "test_dir": "test",
                "image_size": 32,
                "num_classes": 2,
            },
            "training": {"batch_size": 2, "num_workers": 0},
            "augmentation": {
                "train": {
                    "rotate_limit": 10,
                    "rotate_p": 0.5,
                    "h_flip_p": 0.5,
                    "v_flip_p": 0.5,
                    "brightness_limit": 0.1,
                    "contrast_limit": 0.1,
                    "brightness_contrast_p": 0.1,
                    "gamma_limit": [95, 105],
                    "gamma_p": 0.1,
                },
                "normalize": {
                    "mean": [0.0, 0.0, 0.0],
                    "std": [1.0, 1.0, 1.0],
                    "max_pixel_value": 255.0,
                },
            },
        }

        train_loader, val_loader, test_loader, _ = create_dataloaders(config, distributed=False)

        assert test_loader is not None
        for loader in (train_loader, val_loader, test_loader):
            images, masks = next(iter(loader))
            assert images.shape == (2, 3, 32, 32)
            assert masks.shape == (2, 32, 32)
