"""
Data Module - DataLoader Creation
"""
import os

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, DistributedSampler

from .dataset import SegmentationDataset, get_image_mask_pairs
from .transforms import get_train_transforms, get_val_transforms


def create_dataloaders(
    config: dict, distributed: bool = False, world_size: int = 1, rank: int = 0
) -> tuple:
    """
    Create train, validation, and test dataloaders.

    Args:
        config: Configuration dictionary
        distributed: Whether to use distributed training
        world_size: Number of distributed processes
        rank: Current process rank

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_cfg = config["data"]
    train_cfg = config["training"]
    aug_cfg = config["augmentation"]

    # Build paths
    base_dir = data_cfg["base_dir"]
    dataset_dir = os.path.join(base_dir, data_cfg["dataset_dir"])

    train_dir = os.path.join(dataset_dir, data_cfg["train_dir"])
    val_dir = os.path.join(dataset_dir, data_cfg["val_dir"])
    test_dir = os.path.join(dataset_dir, data_cfg["test_dir"])

    # Get transforms
    train_transform = get_train_transforms(
        height=data_cfg["image_size"],
        width=data_cfg["image_size"],
        rotate_limit=aug_cfg["train"]["rotate_limit"],
        rotate_p=aug_cfg["train"]["rotate_p"],
        h_flip_p=aug_cfg["train"]["h_flip_p"],
        v_flip_p=aug_cfg["train"]["v_flip_p"],
        brightness_limit=aug_cfg["train"]["brightness_limit"],
        contrast_limit=aug_cfg["train"]["contrast_limit"],
        brightness_contrast_p=aug_cfg["train"]["brightness_contrast_p"],
        gamma_limit=tuple(aug_cfg["train"]["gamma_limit"]),
        gamma_p=aug_cfg["train"]["gamma_p"],
        mean=aug_cfg["normalize"]["mean"],
        std=aug_cfg["normalize"]["std"],
        max_pixel_value=aug_cfg["normalize"]["max_pixel_value"],
    )

    val_transform = get_val_transforms(
        height=data_cfg["image_size"],
        width=data_cfg["image_size"],
        mean=aug_cfg["normalize"]["mean"],
        std=aug_cfg["normalize"]["std"],
        max_pixel_value=aug_cfg["normalize"]["max_pixel_value"],
    )

    # Get image/mask pairs for training
    train_images_dir = os.path.join(train_dir, "images")
    train_masks_dir = os.path.join(train_dir, "masks")
    train_images, train_masks = get_image_mask_pairs(train_images_dir, train_masks_dir)

    # Split training data into train/val if needed
    if os.path.exists(val_dir):
        val_images_dir = os.path.join(val_dir, "images")
        val_masks_dir = os.path.join(val_dir, "masks")
        val_images, val_masks = get_image_mask_pairs(val_images_dir, val_masks_dir)
    else:
        # Create validation split from training data
        train_images, val_images, train_masks, val_masks = train_test_split(
            train_images,
            train_masks,
            test_size=data_cfg["val_split"],
            random_state=config["experiment"]["seed"],
        )

    # Get optimization settings
    use_onehot = data_cfg.get("use_onehot_encoding", False)  # Default: optimized class indices
    cache_val = data_cfg.get("cache_validation", False)  # Cache validation set in memory

    # Create datasets
    train_dataset = SegmentationDataset(
        train_images,
        train_masks,
        transform=train_transform,
        num_classes=data_cfg["num_classes"],
        cache_in_memory=False,  # Don't cache training (augmentation varies)
        use_onehot=use_onehot,
    )

    val_dataset = SegmentationDataset(
        val_images,
        val_masks,
        transform=val_transform,
        num_classes=data_cfg["num_classes"],
        cache_in_memory=cache_val,  # Cache validation for speedup
        use_onehot=use_onehot,
    )

    # Get DataLoader optimization settings
    num_workers = train_cfg.get("num_workers", 8)
    prefetch_factor = train_cfg.get("prefetch_factor", 2)
    persistent_workers = train_cfg.get("persistent_workers", True) and num_workers > 0

    # Test dataset (if exists)
    test_loader = None
    if os.path.exists(test_dir):
        test_images_dir = os.path.join(test_dir, "images")
        test_masks_dir = os.path.join(test_dir, "masks")
        if os.path.exists(test_images_dir) and os.path.exists(test_masks_dir):
            test_images, test_masks = get_image_mask_pairs(test_images_dir, test_masks_dir)
            test_dataset = SegmentationDataset(
                test_images,
                test_masks,
                transform=val_transform,
                num_classes=data_cfg["num_classes"],
                cache_in_memory=False,
                use_onehot=use_onehot,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=train_cfg["batch_size"],
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=persistent_workers,
            )

    # Create samplers for distributed training
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, test_loader, train_sampler
