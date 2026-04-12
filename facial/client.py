"""
Client script to test the Facial Landmark Detection API.
"""

import requests
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

# Configuration
API_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def get_landmarks():
    """Get available landmarks."""
    print("Getting available landmarks...")
    response = requests.get(f"{API_URL}/landmarks")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def predict_image(image_path: str):
    """
    Send an image to the API and get landmark predictions.
    
    Args:
        image_path: Path to the image file
    """
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"Predicting landmarks for: {image_path}")
    
    # Send request
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        print(f"Message: {result['message']}")
        print(f"\nDetected landmarks:")
        
        for landmark in result['landmarks']:
            print(f"  {landmark['id']:2d}. {landmark['label']:15s} -> ({landmark['x']:.2f}, {landmark['y']:.2f})")
        
        # Visualize results
        visualize_landmarks(image_path, result['landmarks'])
    else:
        print(f"Error: {response.text}")

def visualize_landmarks(image_path: str, landmarks: list):
    """
    Visualize detected landmarks on the image.
    
    Args:
        image_path: Path to the original image
        landmarks: List of detected landmarks
    """
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Cannot load image for visualization")
        return
    
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img_rgb)
    ax.set_title(f"Facial Landmarks - {Path(image_path).name}", fontsize=14)
    
    # Draw landmarks
    for landmark in landmarks:
        x, y = landmark['x'], landmark['y']
        label = landmark['label']
        
        ax.scatter(x, y, c='lime', marker='o', s=100, edgecolors='red', linewidths=2, zorder=5)
        ax.text(
            x + 10, y - 10,
            f"{landmark['id']}: {label}",
            color='lime', fontsize=10, fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.7, pad=3)
        )
    
    ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=== Facial Landmark Detection API Client ===\n")
    
    # Test health endpoint
    test_health()
    
    # Get available landmarks
    get_landmarks()
    
    # Test prediction with an image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default test image
        image_path = "data/test/nhat.png"
    
    predict_image(image_path)
