"""
Data Augmentation Transforms
"""
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    height: int = 224,
    width: int = 224,
    rotate_limit: int = 360,
    rotate_p: float = 0.5,
    h_flip_p: float = 0.5,
    v_flip_p: float = 0.5,
    brightness_limit: float = 0.3,
    contrast_limit: float = 0.3,
    brightness_contrast_p: float = 0.3,
    gamma_limit: tuple = (85, 115),
    gamma_p: float = 0.3,
    mean: list = None,
    std: list = None,
    max_pixel_value: float = 255.0,
):
    """
    Get training data augmentation transforms.

    Args:
        height: Target image height
        width: Target image width
        rotate_limit: Maximum rotation angle
        rotate_p: Probability of rotation
        h_flip_p: Probability of horizontal flip
        v_flip_p: Probability of vertical flip
        brightness_limit: Brightness adjustment range
        contrast_limit: Contrast adjustment range
        brightness_contrast_p: Probability of brightness/contrast adjustment
        gamma_limit: Gamma adjustment range (min, max)
        gamma_p: Probability of gamma adjustment
        mean: Normalization mean
        std: Normalization std
        max_pixel_value: Maximum pixel value for normalization

    Returns:
        Albumentations Compose transform
    """
    if mean is None:
        mean = [0.0, 0.0, 0.0]
    if std is None:
        std = [1.0, 1.0, 1.0]

    return A.Compose(
        [
            A.Resize(height=height, width=width, interpolation=cv2.INTER_NEAREST),
            A.Rotate(limit=rotate_limit, p=rotate_p),
            A.HorizontalFlip(p=h_flip_p),
            A.VerticalFlip(p=v_flip_p),
            A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value),
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=brightness_contrast_p,
            ),
            A.RandomGamma(gamma_limit=gamma_limit, p=gamma_p),
            ToTensorV2(),
        ]
    )


def get_val_transforms(
    height: int = 224,
    width: int = 224,
    mean: list = None,
    std: list = None,
    max_pixel_value: float = 255.0,
):
    """
    Get validation/test data transforms.

    Args:
        height: Target image height
        width: Target image width
        mean: Normalization mean
        std: Normalization std
        max_pixel_value: Maximum pixel value for normalization

    Returns:
        Albumentations Compose transform
    """
    if mean is None:
        mean = [0.0, 0.0, 0.0]
    if std is None:
        std = [1.0, 1.0, 1.0]

    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value),
            ToTensorV2(),
        ]
    )
