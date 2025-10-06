"""
Full Dataset Training Script for Cephalometric Landmark Detection
Optimized for RTX 5060 Ti (16GB VRAM)

Features:
- Train on full 700 images
- Wing Loss function
- Mixed Precision Training (FP16)
- Learning Rate Warmup + Cosine Annealing
- Early Stopping
- TensorBoard Logging
- Checkpoint Saving
- Comprehensive Metrics (MRE, SDR@2/2.5/3/4mm)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path

from dataset import CephDataset
from model import LandmarkModel
from losses import WingLoss, get_loss_function
from utils import (
    calculate_mre, calculate_sdr, calculate_per_landmark_error,
    MetricsTracker, EarlyStopping, save_checkpoint, load_checkpoint,
    print_per_landmark_errors
)


class Config:
    """Training Configuration - Optimized for RTX 5060 Ti (16GB)"""

    # Data
    DATA_DIR = './data'
    IMG_SIZE = 1024  # Can try 1024 if you want higher accuracy

    # Model
    BACKBONE = 'efficientnet_b3'  # b3 is good balance (12M params)
    # Try 'efficientnet_b4' for even better accuracy (19M params)

    # Training
    BATCH_SIZE = 16  # RTX 5060 Ti can handle this comfortably
    EPOCHS = 100
    NUM_WORKERS = 4  # Adjust based on CPU cores

    # Optimizer
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP = 1.0  # Gradient clipping

    # Learning Rate Schedule
    WARMUP_EPOCHS = 5
    LR_MIN = 1e-6

    # Loss
    LOSS_TYPE = 'wing'  # 'wing', 'adaptive_wing', 'smooth_l1', 'mse', 'combined'
    LOSS_PARAMS = {'omega': 10, 'epsilon': 2}

    # Early Stopping
    PATIENCE = 20
    MIN_DELTA = 0.01  # Minimum improvement in MRE (mm)

    # Checkpoints
    SAVE_DIR = './checkpoints_full'
    SAVE_EVERY = 10  # Save checkpoint every N epochs
    RESUME_FROM = None  # Path to checkpoint to resume from

    # Mixed Precision
    USE_AMP = True  # Automatic Mixed Precision (FP16)

    # TensorBoard
    LOG_DIR = './runs_full'

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, lr_min, last_epoch=-1):
    """Learning rate scheduler with warmup and cosine annealing"""

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return lr_min + (1 - lr_min) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def train_epoch(model, loader, optimizer, criterion, device, scaler=None, use_amp=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(loader)

    pbar = tqdm(loader, desc='Training', leave=False)
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        landmarks = batch['landmarks'].to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, landmarks)

            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, landmarks)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, loader, criterion, device):
    """Validate the model and calculate comprehensive metrics"""
    model.eval()
    total_loss = 0

    all_predictions = []
    all_ground_truths = []
    all_pixel_sizes = []
    all_original_sizes = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating', leave=False):
            images = batch['image'].to(device)
            landmarks = batch['landmarks'].to(device)
            pixel_sizes = batch['pixel_size']
            original_sizes = batch['original_size']

            outputs = model(images)
            loss = criterion(outputs, landmarks)
            total_loss += loss.item()

            # Collect predictions
            all_predictions.append(outputs.cpu().numpy())
            all_ground_truths.append(landmarks.cpu().numpy())
            all_pixel_sizes.append(pixel_sizes.numpy())
            all_original_sizes.append(
                np.stack([original_sizes[0].numpy(), original_sizes[1].numpy()], axis=1)
            )

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)  # (N, 29, 2)
    all_ground_truths = np.concatenate(all_ground_truths, axis=0)
    all_pixel_sizes = np.concatenate(all_pixel_sizes, axis=0)
    all_original_sizes = np.concatenate(all_original_sizes, axis=0)

    # Denormalize predictions to original image coordinates
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

    avg_loss = total_loss / len(loader)

    return {
        'loss': avg_loss,
        'mre': mre,
        'sdr_2mm': sdr_2mm,
        'sdr_2_5mm': sdr_2_5mm,
        'sdr_3mm': sdr_3mm,
        'sdr_4mm': sdr_4mm,
        'per_landmark_errors': per_landmark_errors
    }


def train(config):
    """Main training function"""

    print("="*80)
    print("Cephalometric Landmark Detection - Full Dataset Training")
    print("="*80)
    print(f"Device: {config.DEVICE}")
    print(f"Backbone: {config.BACKBONE}")
    print(f"Image Size: {config.IMG_SIZE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Learning Rate: {config.LR}")
    print(f"Loss Function: {config.LOSS_TYPE}")
    print(f"Mixed Precision: {config.USE_AMP}")
    print("="*80)

    # Create directories
    save_dir = Path(config.SAVE_DIR)
    save_dir.mkdir(exist_ok=True)
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(exist_ok=True)

    # Datasets
    print("\nLoading datasets...")
    train_dataset = CephDataset(config.DATA_DIR, mode='train', img_size=config.IMG_SIZE)
    val_dataset = CephDataset(config.DATA_DIR, mode='valid', img_size=config.IMG_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Model
    print(f"\nInitializing model ({config.BACKBONE})...")
    model = LandmarkModel(num_landmarks=29, backbone=config.BACKBONE)
    model = model.to(config.DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function
    criterion = get_loss_function(config.LOSS_TYPE, **config.LOSS_PARAMS)
    print(f"Loss function: {config.LOSS_TYPE}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )

    # Learning rate scheduler
    scheduler = get_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=config.WARMUP_EPOCHS,
        total_epochs=config.EPOCHS,
        lr_min=config.LR_MIN / config.LR
    )

    # Mixed precision scaler
    scaler = GradScaler() if config.USE_AMP else None

    # Metrics tracker
    metrics_tracker = MetricsTracker(save_dir=save_dir)

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.PATIENCE,
        mode='min',
        min_delta=config.MIN_DELTA
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_mre = float('inf')

    if config.RESUME_FROM:
        start_epoch, best_mre = load_checkpoint(
            model, config.RESUME_FROM,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.DEVICE
        )

    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)

    for epoch in range(start_epoch, config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print("-" * 80)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            config.DEVICE, scaler, config.USE_AMP
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, config.DEVICE)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_metrics['loss']:.4f}")
        print(f"Val MRE:    {val_metrics['mre']:.2f} mm")
        print(f"SDR@2.0mm:  {val_metrics['sdr_2mm']:.2f}%")
        print(f"SDR@2.5mm:  {val_metrics['sdr_2_5mm']:.2f}%")
        print(f"SDR@3.0mm:  {val_metrics['sdr_3mm']:.2f}%")
        print(f"SDR@4.0mm:  {val_metrics['sdr_4mm']:.2f}%")
        print(f"LR:         {current_lr:.2e}")

        # Update metrics tracker
        metrics_tracker.update(
            epoch=epoch+1,
            train_loss=train_loss,
            val_loss=val_metrics['loss'],
            val_mre=val_metrics['mre'],
            val_sdr_2mm=val_metrics['sdr_2mm'],
            val_sdr_2_5mm=val_metrics['sdr_2_5mm'],
            val_sdr_3mm=val_metrics['sdr_3mm'],
            val_sdr_4mm=val_metrics['sdr_4mm'],
            lr=current_lr
        )

        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Metrics/MRE', val_metrics['mre'], epoch)
        writer.add_scalar('Metrics/SDR_2mm', val_metrics['sdr_2mm'], epoch)
        writer.add_scalar('Metrics/SDR_2.5mm', val_metrics['sdr_2_5mm'], epoch)
        writer.add_scalar('Metrics/SDR_3mm', val_metrics['sdr_3mm'], epoch)
        writer.add_scalar('Metrics/SDR_4mm', val_metrics['sdr_4mm'], epoch)
        writer.add_scalar('LR', current_lr, epoch)

        # Save best model
        if val_metrics['mre'] < best_mre:
            best_mre = val_metrics['mre']
            save_checkpoint(
                model, optimizer, scheduler,
                epoch, val_metrics['mre'],
                save_dir / 'best_model_full.pth',
                train_loss=train_loss,
                val_metrics=val_metrics
            )
            print(f"✓ Saved best model (MRE: {best_mre:.2f}mm)")

        # Save periodic checkpoint
        if (epoch + 1) % config.SAVE_EVERY == 0:
            save_checkpoint(
                model, optimizer, scheduler,
                epoch, val_metrics['mre'],
                save_dir / f'checkpoint_epoch_{epoch+1}.pth',
                train_loss=train_loss,
                val_metrics=val_metrics
            )

        # Early stopping check
        improved = early_stopping(val_metrics['mre'])
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best MRE: {best_mre:.2f}mm")
            break

        # Print per-landmark errors for best epoch
        if improved:
            print_per_landmark_errors(val_metrics['per_landmark_errors'])

    # Training completed
    print("\n" + "="*80)
    print("Training Completed!")
    print("="*80)
    print(f"Best MRE: {best_mre:.2f}mm")
    print(f"Best model saved to: {save_dir / 'best_model_full.pth'}")
    print(f"Training log saved to: {save_dir / 'training_log.csv'}")

    # Plot final metrics
    metrics_tracker.plot_metrics(save_dir / 'training_metrics.png')

    # Close TensorBoard writer
    writer.close()

    # Final validation on best model
    print("\nRunning final validation on best model...")
    checkpoint = torch.load(save_dir / 'best_model_full.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    final_metrics = validate(model, val_loader, criterion, config.DEVICE)

    print("\n" + "="*80)
    print("Final Validation Metrics")
    print("="*80)
    print(f"MRE:       {final_metrics['mre']:.2f} mm")
    print(f"SDR@2.0mm: {final_metrics['sdr_2mm']:.2f}%")
    print(f"SDR@2.5mm: {final_metrics['sdr_2_5mm']:.2f}%")
    print(f"SDR@3.0mm: {final_metrics['sdr_3mm']:.2f}%")
    print(f"SDR@4.0mm: {final_metrics['sdr_4mm']:.2f}%")
    print("="*80)

    print_per_landmark_errors(final_metrics['per_landmark_errors'])

    return model


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train cephalometric landmark detection model')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--backbone', type=str, default='efficientnet_b3', help='Backbone model')
    parser.add_argument('--img-size', type=int, default=512, help='Image size')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')

    args = parser.parse_args()

    # Update config from args
    Config.DATA_DIR = args.data_dir
    Config.BATCH_SIZE = args.batch_size
    Config.EPOCHS = args.epochs
    Config.LR = args.lr
    Config.BACKBONE = args.backbone
    Config.IMG_SIZE = args.img_size
    Config.RESUME_FROM = args.resume
    Config.USE_AMP = not args.no_amp

    # Train
    train(Config)


if __name__ == '__main__':
    main()
