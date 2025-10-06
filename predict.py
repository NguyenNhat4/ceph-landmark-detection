import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import LandmarkModel
import json
import argparse

# Tên các landmarks (29 điểm)
LANDMARK_NAMES = [
    "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R", "S",
    "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
    "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N'", "Pog'", "Sn"
]

def load_model(model_path, device='cuda'):
    """Load trained model"""
    model = LandmarkModel(num_landmarks=29, backbone='efficientnet_b0')

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    print(f"✓ Loaded model from {model_path}")
    print(f"  Trained epoch: {checkpoint['epoch']}")
    print(f"  Val MRE: {checkpoint['val_mre']:.2f}mm")

    return model

def preprocess_image(image_path, img_size=512):
    """Preprocess image for model input"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # Resize
    img_resized = cv2.resize(image, (img_size, img_size))

    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = (img_normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    # To tensor
    img_tensor = torch.FloatTensor(img_normalized).permute(2, 0, 1).unsqueeze(0)

    return img_tensor, image, (h, w)

def predict_landmarks(model, image_path, device='cuda'):
    """Predict landmarks on an image"""
    # Preprocess
    img_tensor, original_image, (h, w) = preprocess_image(image_path)

    # Predict
    with torch.no_grad():
        pred = model(img_tensor.to(device))

    # Denormalize landmarks (chuyển về tọa độ ảnh gốc)
    landmarks = pred[0].cpu().numpy() * [w, h]

    return landmarks, original_image

def visualize_prediction(image, landmarks, save_path=None, show_labels=True):
    """Visualize predicted landmarks on image"""
    plt.figure(figsize=(12, 12))
    plt.imshow(image)

    # Plot landmarks
    plt.scatter(landmarks[:, 0], landmarks[:, 1],
               c='red', s=100, alpha=0.7, edgecolors='white', linewidths=2)

    # Annotate với tên landmarks
    if show_labels:
        for i, (x, y) in enumerate(landmarks):
            plt.text(x + 20, y - 20, LANDMARK_NAMES[i],
                    color='yellow', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))

    plt.title('Predicted Cephalometric Landmarks', fontsize=16, fontweight='bold')
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")

    plt.show()

def compare_with_ground_truth(image_path, predicted_landmarks, anno_path=None):
    """So sánh prediction với ground truth (nếu có)"""
    if anno_path is None:
        print("No ground truth provided")
        return

    # Load ground truth
    with open(anno_path, 'r') as f:
        anno = json.load(f)

    gt_landmarks = []
    for lm in anno['landmarks']:
        gt_landmarks.append([lm['value']['x'], lm['value']['y']])
    gt_landmarks = np.array(gt_landmarks)

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calculate errors
    distances = np.sqrt(np.sum((predicted_landmarks - gt_landmarks)**2, axis=1))
    mean_error = np.mean(distances)

    # Visualize comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Ground Truth
    axes[0].imshow(image)
    axes[0].scatter(gt_landmarks[:, 0], gt_landmarks[:, 1],
                   c='green', s=100, alpha=0.7, label='Ground Truth')
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Prediction
    axes[1].imshow(image)
    axes[1].scatter(predicted_landmarks[:, 0], predicted_landmarks[:, 1],
                   c='red', s=100, alpha=0.7, label='Prediction')
    axes[1].set_title('Prediction', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(image)
    axes[2].scatter(gt_landmarks[:, 0], gt_landmarks[:, 1],
                   c='green', s=100, alpha=0.5, label='Ground Truth')
    axes[2].scatter(predicted_landmarks[:, 0], predicted_landmarks[:, 1],
                   c='red', s=100, alpha=0.5, label='Prediction')

    # Draw error lines
    for i in range(len(gt_landmarks)):
        axes[2].plot([gt_landmarks[i, 0], predicted_landmarks[i, 0]],
                    [gt_landmarks[i, 1], predicted_landmarks[i, 1]],
                    'yellow', linewidth=1, alpha=0.5)

    axes[2].set_title(f'Overlay (Mean Error: {mean_error:.2f} pixels)',
                     fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison to comparison.png")
    plt.show()

    # Print per-landmark errors
    print("\nPer-Landmark Errors:")
    print("-" * 50)
    for i, (name, dist) in enumerate(zip(LANDMARK_NAMES, distances)):
        print(f"{name:10s}: {dist:6.2f} pixels")
    print("-" * 50)
    print(f"Mean Error: {mean_error:.2f} pixels")

def save_predictions_json(landmarks, save_path='prediction.json'):
    """Save predictions to JSON file"""
    result = {
        "landmarks": []
    }

    for i, (x, y) in enumerate(landmarks):
        result["landmarks"].append({
            "symbol": LANDMARK_NAMES[i],
            "index": i,
            "coordinates": {
                "x": float(x),
                "y": float(y)
            }
        })

    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"✓ Saved predictions to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Predict cephalometric landmarks')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='best_model_50.pth', help='Path to model checkpoint')
    parser.add_argument('--anno', type=str, default=None, help='Path to ground truth annotation (optional)')
    parser.add_argument('--output', type=str, default='prediction.png', help='Path to save visualization')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--no-labels', action='store_true', help='Hide landmark labels')

    args = parser.parse_args()

    # Check device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.model, device=device)

    # Predict
    print(f"\nPredicting on: {args.image}")
    landmarks, image = predict_landmarks(model, args.image, device=device)

    # Visualize
    visualize_prediction(image, landmarks, save_path=args.output,
                        show_labels=not args.no_labels)

    # Save JSON
    save_predictions_json(landmarks, save_path='prediction.json')

    # Compare with ground truth if provided
    if args.anno:
        print("\nComparing with ground truth...")
        compare_with_ground_truth(args.image, landmarks, args.anno)

if __name__ == '__main__':
    main()
