#!/usr/bin/env python3
"""
Evaluation Script for Semantic Segmentation

Usage:
    python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_model.pth
"""
import os
import sys
import argparse
import json
from datetime import datetime

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import create_dataloaders
from src.models import create_model
from src.utils import SegmentationMetrics


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate(
    model,
    test_loader,
    device: torch.device,
    num_classes: int = 2,
    save_predictions: bool = False,
    output_dir: str = None
) -> dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        num_classes: Number of classes
        save_predictions: Whether to save prediction masks
        output_dir: Output directory for predictions
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    metrics = SegmentationMetrics(num_classes=num_classes, device=device)
    
    all_results = []
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # Update metrics
            metrics.update(outputs, masks)
            
            # Save predictions if requested
            if save_predictions and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                for i in range(predictions.size(0)):
                    idx = batch_idx * test_loader.batch_size + i
                    pred_mask = predictions[i].cpu().numpy().astype(np.uint8) * 255
                    
                    pred_img = Image.fromarray(pred_mask)
                    pred_img.save(os.path.join(output_dir, f'pred_{idx:04d}.png'))
    
    # Compute final metrics
    final_metrics = metrics.compute()
    
    return final_metrics


def compute_confusion_matrix(model, test_loader, device, num_classes=2):
    """
    Compute confusion matrix.
    
    Returns:
        Confusion matrix as numpy array
    """
    model.eval()
    
    confusion_matrix = torch.zeros(num_classes, num_classes, device=device)
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Computing confusion matrix"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # Convert masks to class indices
            if masks.dim() == 4:
                masks = torch.argmax(masks, dim=1)
            
            # Flatten
            predictions = predictions.view(-1)
            masks = masks.view(-1)
            
            # Update confusion matrix
            for t, p in zip(masks.view(-1), predictions.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    return confusion_matrix.cpu().numpy()


def main(args):
    """Main evaluation function."""
    
    # Load configuration
    config = load_config(args.config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create dataloaders
    _, _, test_loader, _ = create_dataloaders(config, distributed=False)
    
    if test_loader is None:
        print("No test dataset found. Using validation set for evaluation.")
        _, test_loader, _, _ = create_dataloaders(config, distributed=False)
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model_cfg = config['model']
    model = create_model(
        architecture=model_cfg['architecture'],
        encoder=model_cfg['encoder'],
        encoder_weights=None,  # Don't load pretrained weights
        num_classes=model_cfg['num_classes'],
        activation=model_cfg['activation']
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    
    # Evaluate
    output_dir = None
    if args.save_predictions:
        output_dir = os.path.join(
            config['paths']['output_dir'],
            'predictions',
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )
    
    metrics = evaluate(
        model, test_loader, device,
        num_classes=config['data']['num_classes'],
        save_predictions=args.save_predictions,
        output_dir=output_dir
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("=" * 50)
    
    # Compute confusion matrix
    if args.confusion_matrix:
        cm = compute_confusion_matrix(
            model, test_loader, device,
            num_classes=config['data']['num_classes']
        )
        print("\nConfusion Matrix:")
        print(cm)
    
    # Save results
    if args.output:
        results = {
            'checkpoint': args.checkpoint,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate semantic segmentation model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save evaluation results (JSON)'
    )
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save prediction masks'
    )
    parser.add_argument(
        '--confusion-matrix',
        action='store_true',
        help='Compute and print confusion matrix'
    )
    
    args = parser.parse_args()
    main(args)
