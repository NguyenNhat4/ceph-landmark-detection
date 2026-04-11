import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import CephDataset
from predict import load_model, LANDMARK_NAMES
from utils import (
    calculate_mre, calculate_sdr, calculate_per_landmark_error,
    print_per_landmark_errors
)

def test_evaluation(model, loader, device, save_dir):
    """Evaluate on the test set and generate charts"""
    model.eval()

    all_predictions = []
    all_ground_truths = []
    all_pixel_sizes = []
    all_original_sizes = []
    all_images = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Testing'):
            images = batch['image'].to(device)
            landmarks = batch['landmarks'].to(device)
            pixel_sizes = batch['pixel_size']
            original_sizes = batch['original_size']

            outputs = model(images)
            
            # Store CPU tensors
            all_predictions.append(outputs.cpu().numpy())
            all_ground_truths.append(landmarks.cpu().numpy())
            all_pixel_sizes.append(pixel_sizes.numpy())
            all_original_sizes.append(
                np.stack([original_sizes[0].numpy(), original_sizes[1].numpy()], axis=1)
            )
            # Store some images for plotting (limit to first batch to save memory)
            if len(all_images) < 16:
                all_images.append(images.cpu().numpy())

    # Concatenate everything
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truths = np.concatenate(all_ground_truths, axis=0)
    all_pixel_sizes = np.concatenate(all_pixel_sizes, axis=0)
    all_original_sizes = np.concatenate(all_original_sizes, axis=0)
    img_samples = np.concatenate(all_images, axis=0)

    # Denormalize landmarks to original scale
    for i in range(len(all_predictions)):
        h, w = all_original_sizes[i]
        all_predictions[i] = all_predictions[i] * [w, h]
        all_ground_truths[i] = all_ground_truths[i] * [w, h]

    # Calculate metrics
    avg_pixel_size = np.mean(all_pixel_sizes)
    mre = calculate_mre(all_predictions, all_ground_truths, avg_pixel_size)
    sdr_2mm = calculate_sdr(all_predictions, all_ground_truths, 2.0, avg_pixel_size)
    sdr_2_5mm = calculate_sdr(all_predictions, all_ground_truths, 2.5, avg_pixel_size)
    sdr_3mm = calculate_sdr(all_predictions, all_ground_truths, 3.0, avg_pixel_size)
    sdr_4mm = calculate_sdr(all_predictions, all_ground_truths, 4.0, avg_pixel_size)

    per_landmark_errors = calculate_per_landmark_error(
        all_predictions, all_ground_truths, avg_pixel_size
    )

    print("\n" + "="*80)
    print("Final Test Metrics")
    print("="*80)
    print(f"Test MRE:     {mre:.2f} mm")
    print(f"SDR@2.0mm:    {sdr_2mm:.2f}%")
    print(f"SDR@2.5mm:    {sdr_2_5mm:.2f}%")
    print(f"SDR@3.0mm:    {sdr_3mm:.2f}%")
    print(f"SDR@4.0mm:    {sdr_4mm:.2f}%")
    print("="*80)

    print_per_landmark_errors(per_landmark_errors, LANDMARK_NAMES)

    # 1. Bar Chart of Per-Landmark Errors
    plot_landmark_errors(per_landmark_errors, LANDMARK_NAMES, save_dir)
    
    # 2. Visualize a few samples
    plot_predictions(img_samples, all_predictions, all_ground_truths, all_original_sizes, save_dir)


def plot_landmark_errors(errors, names, save_dir):
    plt.figure(figsize=(15, 8))
    x_pos = np.arange(len(names))
    
    plt.bar(x_pos, errors, color='skyblue', edgecolor='black')
    plt.axhline(y=np.mean(errors), color='r', linestyle='--', label=f'Mean Error ({np.mean(errors):.2f}mm)')
    
    plt.xticks(x_pos, names, rotation=45, ha='right', fontsize=10)
    plt.ylabel('Mean Radial Error (mm)', fontsize=12)
    plt.title('Per-Landmark Prediction Error on Test Set', fontsize=16, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    save_path = save_dir / 'test_landmark_errors_chart.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved per-landmark errors chart to {save_path}")
    plt.close()


def plot_predictions(images, preds, gts, original_sizes, save_dir, num_samples=4):
    num_samples = min(num_samples, len(images))
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(16, 8 * num_samples))
    if num_samples == 1: axes = [axes]
    
    for idx, sample_idx in enumerate(indices):
        # Decode image
        img = images[sample_idx].transpose(1, 2, 0)
        img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        img = np.clip(img, 0, 1)
        h, w = original_sizes[sample_idx]
        
        # Scale back predictions and ground truth to the 512x512 image tensor resolution
        pred_coords = preds[sample_idx] * [img.shape[1]/w, img.shape[0]/h]
        gt_coords = gts[sample_idx] * [img.shape[1]/w, img.shape[0]/h]
        
        # Plot GT
        ax_gt = axes[idx][0]
        ax_gt.imshow(img)
        ax_gt.scatter(gt_coords[:, 0], gt_coords[:, 1], c='green', s=40, label='Ground Truth')
        ax_gt.set_title("Ground Truth", fontsize=14)
        ax_gt.axis('off')
        
        # Plot Pred vs GT
        ax_pred = axes[idx][1]
        ax_pred.imshow(img)
        ax_pred.scatter(gt_coords[:, 0], gt_coords[:, 1], c='green', s=40, alpha=0.5, label='Ground Truth')
        ax_pred.scatter(pred_coords[:, 0], pred_coords[:, 1], c='red', s=40, alpha=0.7, label='Prediction')
        
        # Draw error lines
        for i in range(len(gt_coords)):
            ax_pred.plot([gt_coords[i, 0], pred_coords[i, 0]],
                         [gt_coords[i, 1], pred_coords[i, 1]],
                         'yellow', linewidth=1, alpha=0.5)
            
        ax_pred.set_title("Prediction vs Ground Truth", fontsize=14)
        ax_pred.legend()
        ax_pred.axis('off')
        
    plt.tight_layout()
    save_path = save_dir / 'test_visual_chart.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved test visual samples chart to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test cephalometric landmark detection model')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model', type=str, default='checkpoints_full/best_model_full.pth', help='Path to model checkpoint')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone model (resnet50, hrnet, etc)')
    parser.add_argument('--hrnet_config', type=str, default='models/configs/hrnet.yaml', help='Path to hrnet config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='runs_full/test_results', help='Directory to save outputs')
    
    args = parser.parse_args()

    # Check device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Save Dir
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    test_dataset = CephDataset(args.data_dir, mode='test', img_size=512)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Loaded test dataset: {len(test_dataset)} images")

    # Load model
    print(f"\nInitializing model ({args.backbone})...")
    model = load_model(args.model, device=device, backbone=args.backbone, hrnet_config=args.hrnet_config)

    test_evaluation(model, test_loader, device, save_dir)

if __name__ == '__main__':
    main()
