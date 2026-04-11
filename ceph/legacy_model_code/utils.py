import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from datetime import datetime

def calculate_mre(pred_landmarks, gt_landmarks, pixel_size=0.1):
    """
    Calculate Mean Radial Error in millimeters

    Args:
        pred_landmarks: (N, 29, 2) or (29, 2) predicted coordinates
        gt_landmarks: (N, 29, 2) or (29, 2) ground truth coordinates
        pixel_size: float or array, pixel to mm conversion

    Returns:
        mre_mm: Mean Radial Error in millimeters
    """
    # Handle single sample
    if pred_landmarks.ndim == 2:
        pred_landmarks = pred_landmarks[np.newaxis, ...]
        gt_landmarks = gt_landmarks[np.newaxis, ...]

    # Calculate Euclidean distances
    distances = np.sqrt(np.sum((pred_landmarks - gt_landmarks)**2, axis=-1))  # (N, 29)

    # Convert to mm
    if isinstance(pixel_size, (int, float)):
        distances_mm = distances * pixel_size
    else:
        distances_mm = distances * pixel_size[:, np.newaxis]

    # Mean across landmarks and batch
    mre_mm = np.mean(distances_mm)

    return mre_mm


def calculate_sdr(pred_landmarks, gt_landmarks, threshold_mm=2.0, pixel_size=0.1):
    """
    Calculate Successful Detection Rate

    Args:
        pred_landmarks: (N, 29, 2) predicted coordinates
        gt_landmarks: (N, 29, 2) ground truth coordinates
        threshold_mm: distance threshold in millimeters
        pixel_size: pixel to mm conversion

    Returns:
        sdr: Percentage of landmarks within threshold
    """
    # Calculate distances
    distances = np.sqrt(np.sum((pred_landmarks - gt_landmarks)**2, axis=-1))

    # Convert to mm
    if isinstance(pixel_size, (int, float)):
        distances_mm = distances * pixel_size
    else:
        distances_mm = distances * pixel_size[:, np.newaxis]

    # Calculate SDR
    sdr = np.mean(distances_mm < threshold_mm) * 100

    return sdr


def calculate_per_landmark_error(pred_landmarks, gt_landmarks, pixel_size=0.1):
    """
    Calculate error for each landmark

    Args:
        pred_landmarks: (N, 29, 2) predicted coordinates
        gt_landmarks: (N, 29, 2) ground truth coordinates
        pixel_size: pixel to mm conversion

    Returns:
        errors: (29,) array of mean errors per landmark in mm
    """
    # Calculate distances per landmark
    distances = np.sqrt(np.sum((pred_landmarks - gt_landmarks)**2, axis=-1))  # (N, 29)

    # Convert to mm
    if isinstance(pixel_size, (int, float)):
        distances_mm = distances * pixel_size
    else:
        distances_mm = distances * pixel_size[:, np.newaxis]

    # Mean across batch
    per_landmark_errors = np.mean(distances_mm, axis=0)  # (29,)

    return per_landmark_errors


class MetricsTracker:
    """Track and save training metrics"""

    def __init__(self, save_dir='./logs'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_mre': [],
            'val_sdr_2mm': [],
            'val_sdr_2_5mm': [],
            'val_sdr_3mm': [],
            'val_sdr_4mm': [],
            'lr': []
        }

        # CSV file path
        self.csv_path = self.save_dir / 'training_log.csv'

        # Initialize CSV
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.metrics.keys())

    def update(self, **kwargs):
        """Update metrics"""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)

        # Append to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [kwargs.get(key, '') for key in self.metrics.keys()]
            writer.writerow(row)

    def plot_metrics(self, save_path=None):
        """Plot training metrics"""
        if len(self.metrics['epoch']) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(self.metrics['epoch'], self.metrics['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.metrics['epoch'], self.metrics['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # MRE
        axes[0, 1].plot(self.metrics['epoch'], self.metrics['val_mre'], label='Val MRE', color='green')
        axes[0, 1].axhline(y=2.0, color='r', linestyle='--', label='Clinical Threshold (2mm)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MRE (mm)')
        axes[0, 1].set_title('Validation MRE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # SDR
        axes[1, 0].plot(self.metrics['epoch'], self.metrics['val_sdr_2mm'], label='SDR@2mm')
        axes[1, 0].plot(self.metrics['epoch'], self.metrics['val_sdr_2_5mm'], label='SDR@2.5mm')
        axes[1, 0].plot(self.metrics['epoch'], self.metrics['val_sdr_3mm'], label='SDR@3mm')
        axes[1, 0].plot(self.metrics['epoch'], self.metrics['val_sdr_4mm'], label='SDR@4mm')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SDR (%)')
        axes[1, 0].set_title('Successful Detection Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Learning Rate
        axes[1, 1].plot(self.metrics['epoch'], self.metrics['lr'], color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path is None:
            save_path = self.save_dir / 'training_metrics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Metrics plot saved to {save_path}")

    def get_best_epoch(self, metric='val_mre', mode='min'):
        """Get best epoch based on metric"""
        if len(self.metrics[metric]) == 0:
            return None

        if mode == 'min':
            best_idx = np.argmin(self.metrics[metric])
        else:
            best_idx = np.argmax(self.metrics[metric])

        return self.metrics['epoch'][best_idx]


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving"""

    def __init__(self, patience=20, mode='min', min_delta=0.0):
        """
        Args:
            patience: How many epochs to wait after last improvement
            mode: 'min' or 'max'
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        """
        Args:
            score: Current validation metric

        Returns:
            improved: True if improved
        """
        if self.best_score is None:
            self.best_score = score
            return True

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def save_checkpoint(model, optimizer, scheduler, epoch, val_mre, save_path, **kwargs):
    """
    Save training checkpoint

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        val_mre: Validation MRE
        save_path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_mre': val_mre,
        'timestamp': datetime.now().isoformat(),
    }

    # Add additional items
    checkpoint.update(kwargs)

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, device='cuda'):
    """
    Load training checkpoint

    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint
        optimizer: PyTorch optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        device: Device to load on

    Returns:
        start_epoch: Epoch to resume from
        val_mre: Best validation MRE
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint.get('epoch', 0) + 1
    val_mre = checkpoint.get('val_mre', float('inf'))

    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {start_epoch}, best MRE: {val_mre:.2f}mm")

    return start_epoch, val_mre


# Landmark names for reporting
LANDMARK_NAMES = [
    "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R", "S",
    "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
    "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N'", "Pog'", "Sn"
]


def print_per_landmark_errors(errors, landmark_names=LANDMARK_NAMES):
    """Print per-landmark errors in a nice format"""
    print("\n" + "="*60)
    print("Per-Landmark Errors (mm)")
    print("="*60)

    for name, error in zip(landmark_names, errors):
        print(f"{name:10s}: {error:6.2f}mm")

    print("="*60)
    print(f"{'Mean':10s}: {np.mean(errors):6.2f}mm")
    print("="*60)
