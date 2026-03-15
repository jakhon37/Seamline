#!/usr/bin/env python3
"""
Inference Script for Semantic Segmentation

Supports:
- Single image inference
- Batch inference on directory
- Chunked inference for large images

Usage:
    Single image:
        python inference.py --checkpoint model.pth --input image.tif --output mask.tif
    
    Directory:
        python inference.py --checkpoint model.pth --input ./images/ --output ./masks/
    
    Large image with chunking:
        python inference.py --checkpoint model.pth --input large.tif --output mask.tif --chunk
"""
import os
import sys
import argparse
from typing import List, Tuple

import yaml
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import create_model


class ImageChunk:
    """Represents a chunk of an image."""
    
    def __init__(self, image: np.ndarray, position: Tuple[int, int]):
        self.image = image
        self.position = position  # (x, y) of top-left corner
        self.mask = None


def get_transform(image_size: int = 224):
    """Get inference transform."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])


def load_model(
    checkpoint_path: str,
    config_path: str = None,
    device: torch.device = None
) -> torch.nn.Module:
    """
    Load trained model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file (optional)
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Default model config
    architecture = "Unet"
    encoder = "resnet18"
    num_classes = 2
    activation = "softmax2d"
    
    # Override with config if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_cfg = config['model']
        architecture = model_cfg['architecture']
        encoder = model_cfg['encoder']
        num_classes = model_cfg['num_classes']
        activation = model_cfg['activation']
    
    # Create model
    model = create_model(
        architecture=architecture,
        encoder=encoder,
        encoder_weights=None,
        num_classes=num_classes,
        activation=activation
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def predict_single(
    model: torch.nn.Module,
    image: np.ndarray,
    transform,
    device: torch.device,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Predict mask for a single image.
    
    Args:
        model: Trained model
        image: Input image (H, W, C)
        transform: Albumentations transform
        device: Device
        threshold: Threshold for binary prediction
        
    Returns:
        Predicted mask (H, W)
    """
    original_size = image.shape[:2]
    
    # Apply transform
    transformed = transform(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output.squeeze(), dim=0)
    
    # Convert to numpy
    mask = pred.cpu().numpy().astype(np.uint8)
    
    # Resize back to original size
    if mask.shape != original_size:
        mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    
    return mask


def crop_image_into_chunks(
    image: np.ndarray,
    chunk_size: int = 224,
    overlap: int = 64
) -> List[ImageChunk]:
    """
    Crop large image into overlapping chunks.
    
    Args:
        image: Input image (H, W, C) or (H, W)
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of ImageChunk objects
    """
    if len(image.shape) == 2:
        height, width = image.shape
    else:
        height, width = image.shape[:2]
    
    stride = chunk_size - overlap
    
    chunks = []
    y = 0
    while y < height:
        x = 0
        while x < width:
            # Calculate chunk boundaries
            x_end = min(x + chunk_size, width)
            y_end = min(y + chunk_size, height)
            
            # Adjust start if we're at the edge
            x_start = max(0, x_end - chunk_size)
            y_start = max(0, y_end - chunk_size)
            
            # Extract chunk
            chunk_image = image[y_start:y_end, x_start:x_end]
            
            # Pad if necessary
            if chunk_image.shape[0] < chunk_size or chunk_image.shape[1] < chunk_size:
                if len(image.shape) == 3:
                    padded = np.zeros((chunk_size, chunk_size, image.shape[2]), dtype=image.dtype)
                else:
                    padded = np.zeros((chunk_size, chunk_size), dtype=image.dtype)
                padded[:chunk_image.shape[0], :chunk_image.shape[1]] = chunk_image
                chunk_image = padded
            
            chunks.append(ImageChunk(chunk_image, (x_start, y_start)))
            
            x += stride
            if x >= width - overlap:
                break
        
        y += stride
        if y >= height - overlap:
            break
    
    return chunks


def combine_chunk_masks(
    chunks: List[ImageChunk],
    original_size: Tuple[int, int],
    chunk_size: int = 224
) -> np.ndarray:
    """
    Combine chunk masks into full image mask.
    
    Args:
        chunks: List of ImageChunk objects with masks
        original_size: Original image size (H, W)
        chunk_size: Size of each chunk
        
    Returns:
        Combined mask
    """
    height, width = original_size
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    weight_map = np.zeros((height, width), dtype=np.float32)
    
    for chunk in chunks:
        if chunk.mask is None:
            continue
        
        x, y = chunk.position
        mask = chunk.mask
        
        # Calculate actual region to copy
        h = min(chunk_size, height - y)
        w = min(chunk_size, width - x)
        
        # Use maximum to combine overlapping regions
        combined_mask[y:y+h, x:x+w] = np.maximum(
            combined_mask[y:y+h, x:x+w],
            mask[:h, :w]
        )
    
    return combined_mask


def predict_chunked(
    model: torch.nn.Module,
    image: np.ndarray,
    transform,
    device: torch.device,
    chunk_size: int = 224,
    overlap: int = 64,
    batch_size: int = 32
) -> np.ndarray:
    """
    Predict mask for large image using chunking.
    
    Args:
        model: Trained model
        image: Input image
        transform: Transform
        device: Device
        chunk_size: Chunk size
        overlap: Overlap between chunks
        batch_size: Batch size for processing chunks
        
    Returns:
        Predicted mask
    """
    original_size = image.shape[:2]
    
    # Crop into chunks
    chunks = crop_image_into_chunks(image, chunk_size, overlap)
    
    # Process chunks in batches
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        
        # Prepare batch
        batch_images = []
        for chunk in batch_chunks:
            transformed = transform(image=chunk.image)
            batch_images.append(transformed['image'])
        
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(batch_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        # Store results
        for j, chunk in enumerate(batch_chunks):
            chunk.mask = predictions[j].cpu().numpy().astype(np.uint8)
    
    # Combine masks
    combined_mask = combine_chunk_masks(chunks, original_size, chunk_size)
    
    return combined_mask


def process_directory(
    model: torch.nn.Module,
    input_dir: str,
    output_dir: str,
    transform,
    device: torch.device,
    use_chunking: bool = False,
    chunk_size: int = 224,
    extensions: tuple = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
):
    """
    Process all images in a directory.
    
    Args:
        model: Trained model
        input_dir: Input directory
        output_dir: Output directory
        transform: Transform
        device: Device
        use_chunking: Whether to use chunking for large images
        chunk_size: Chunk size (if using chunking)
        extensions: Valid image extensions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(extensions) and not f.endswith('_mask.tif')
    ]
    
    print(f"Found {len(image_files)} images")
    
    for filename in tqdm(image_files, desc="Processing"):
        input_path = os.path.join(input_dir, filename)
        
        # Load image
        image = np.array(Image.open(input_path).convert('RGB'))
        
        # Predict
        if use_chunking:
            mask = predict_chunked(model, image, transform, device, chunk_size)
        else:
            mask = predict_single(model, image, transform, device)
        
        # Save mask
        mask_filename = os.path.splitext(filename)[0] + '_mask.tif'
        output_path = os.path.join(output_dir, mask_filename)
        
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(output_path)


def main(args):
    """Main inference function."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, args.config, device)
    print(f"Loaded model from: {args.checkpoint}")
    
    # Get transform
    transform = get_transform(args.image_size)
    
    # Check if input is file or directory
    if os.path.isfile(args.input):
        # Single image
        print(f"Processing single image: {args.input}")
        
        image = np.array(Image.open(args.input).convert('RGB'))
        
        if args.chunk:
            mask = predict_chunked(
                model, image, transform, device,
                args.chunk_size, args.overlap, args.batch_size
            )
        else:
            mask = predict_single(model, image, transform, device)
        
        # Save
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(args.output)
        print(f"Saved mask to: {args.output}")
        
    elif os.path.isdir(args.input):
        # Directory
        print(f"Processing directory: {args.input}")
        
        process_directory(
            model, args.input, args.output, transform, device,
            use_chunking=args.chunk,
            chunk_size=args.chunk_size
        )
        print(f"Saved masks to: {args.output}")
    
    else:
        raise ValueError(f"Input not found: {args.input}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on segmentation images')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (optional)'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input image or directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output mask or directory'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Model input size'
    )
    parser.add_argument(
        '--chunk',
        action='store_true',
        help='Use chunking for large images'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=224,
        help='Chunk size for large images'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=64,
        help='Overlap between chunks'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for chunk processing'
    )
    
    args = parser.parse_args()
    main(args)
