"""
Semantic Segmentation Dataset
"""
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation tasks."""
    
    def __init__(
        self,
        image_paths: list,
        mask_paths: list,
        transform=None,
        num_classes: int = 2
    ):
        """
        Args:
            image_paths: List of paths to images
            mask_paths: List of paths to masks
            transform: Albumentations transform
            num_classes: Number of segmentation classes
        """
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        self.images = [img for img in image_paths if img.lower().endswith(valid_extensions)]
        self.masks = [mask for mask in mask_paths if mask.lower().endswith(valid_extensions)]
        self.transform = transform
        self.num_classes = num_classes
        
        assert len(self.images) == len(self.masks), \
            f"Number of images ({len(self.images)}) != number of masks ({len(self.masks)})"
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        # Load image
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Load mask
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize(image.shape[1::-1], Image.NEAREST)
        mask = np.array(mask)
        mask = np.clip(mask, 0, self.num_classes - 1)
        
        # Apply transforms
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Convert mask to one-hot encoding
        mask_onehot = np.eye(self.num_classes)[mask]
        mask_onehot = torch.from_numpy(mask_onehot).float()
        mask_onehot = mask_onehot.permute(2, 0, 1)
        
        return image, mask_onehot


def get_image_mask_pairs(images_dir: str, masks_dir: str) -> tuple:
    """
    Get sorted lists of image and mask paths.
    
    Args:
        images_dir: Directory containing images
        masks_dir: Directory containing masks
        
    Returns:
        Tuple of (image_paths, mask_paths)
    """
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    # Get image files
    image_files = [
        f for f in os.listdir(images_dir) 
        if f.lower().endswith(valid_extensions) and not f.endswith('_mask.tif')
    ]
    
    # Get mask files
    mask_files = [
        f for f in os.listdir(masks_dir) 
        if f.lower().endswith(valid_extensions)
    ]
    
    # Sort by base name
    def sorting_key(filename):
        base_name = os.path.splitext(filename)[0]
        return base_name.replace('_mask', '')
    
    image_files.sort(key=sorting_key)
    mask_files.sort(key=sorting_key)
    
    # Create full paths
    image_paths = [os.path.join(images_dir, f) for f in image_files]
    mask_paths = [os.path.join(masks_dir, f) for f in mask_files]
    
    return image_paths, mask_paths
