import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset import CephDataset
from model import LandmarkModel

def calculate_mre(pred, target, pixel_sizes, original_sizes):
    """Calculate MRE in mm"""
    batch_mre = []
    for i in range(len(pred)):
        h, w = original_sizes[i]
        # Denormalize coordinates
        pred_coords = pred[i].cpu().numpy() * [w, h]
        target_coords = target[i].cpu().numpy() * [w, h]

        # Calculate distances
        distances = np.sqrt(np.sum((pred_coords - target_coords)**2, axis=1))
        mre_pixels = np.mean(distances)
        mre_mm = mre_pixels * pixel_sizes[i].item()
        batch_mre.append(mre_mm)

    return np.mean(batch_mre)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        landmarks = batch['landmarks'].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, landmarks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_mre = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating'):
            images = batch['image'].to(device)
            landmarks = batch['landmarks'].to(device)
            pixel_sizes = batch['pixel_size']
            original_sizes = batch['original_size']

            outputs = model(images)
            loss = criterion(outputs, landmarks)
            total_loss += loss.item()

            # Calculate MRE
            mre = calculate_mre(outputs, landmarks, pixel_sizes,
                               list(zip(original_sizes[0].tolist(),
                                       original_sizes[1].tolist())))
            all_mre.append(mre)

    return total_loss / len(loader), np.mean(all_mre)

def main():
    # Config
    DATA_DIR = './data'
    BATCH_SIZE = 4
    EPOCHS = 50
    LR = 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {DEVICE}")

    # Datasets
    train_dataset = CephDataset(DATA_DIR, mode='train', img_size=512)
    val_dataset = CephDataset(DATA_DIR, mode='valid', img_size=512)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Model
    model = LandmarkModel(num_landmarks=29, backbone='efficientnet_b3')
    model = model.to(DEVICE)

    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    best_mre = float('inf')

    # Training loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_mre = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val MRE: {val_mre:.2f}mm")

        # Save best model
        if val_mre < best_mre:
            best_mre = val_mre
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mre': val_mre,
            }, 'best_model.pth')
            print(f"✓ Saved best model (MRE: {best_mre:.2f}mm)")

    print(f"\n🎉 Training completed! Best MRE: {best_mre:.2f}mm")

if __name__ == '__main__':
    main()