# Facial Landmark Detection API

FastAPI-based REST API for detecting facial landmarks using HRNet model trained on cephalometric data.

## 📋 Overview

This API provides a simple interface to run inference on facial images and detect 8 cephalometric landmarks:
- Glabella
- N' (Nasion')
- Pronasal
- Subnasale
- Labiale Superius
- Labiale Inferius
- B' (B point')
- Pog' (Pogonion')

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-api.txt
```

### 2. Start the API Server

```bash
python api.py
```

The API will start at `http://localhost:8000`

### 3. Test the API

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Get Available Landmarks
```bash
curl http://localhost:8000/landmarks
```

#### Predict Landmarks from an Image
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/test/nhat.png"
```

Or use the Python client:
```bash
python client.py data/test/nhat.png
```

## 📚 API Documentation

FastAPIdocs are automatically generated and available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Endpoints

#### `GET /health`
Health check endpoint. Returns model status and device information.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "num_landmarks": 8,
  "model_loaded": true
}
```

#### `GET /landmarks`
Get list of available landmarks.

**Response:**
```json
{
  "landmarks": ["Glabella", "N'", "Pronasal", ...],
  "count": 8
}
```

#### `POST /predict`
Predict facial landmarks from an uploaded image.

**Request:**
- Content-Type: `multipart/form-data`
- Parameter: `file` (image file - JPEG or PNG)

**Response:**
```json
{
  "success": true,
  "landmarks": [
    {
      "id": 0,
      "label": "Glabella",
      "x": 245.5,
      "y": 120.3
    },
    ...
  ],
  "message": "Successfully detected 8 facial landmarks"
}
```

## 🛠️ Configuration

Edit the `Config` class in `api.py` to customize:

```python
class Config:
    MODEL_PATH = "models/hrnet_finetuned_8pts.pth"  # Path to model weights
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU or CPU
    NUM_KEYPOINTS = 8  # Number of landmarks
    IMAGE_SIZE = [512, 512]  # Model input size
    HEATMAP_SIZE = [128, 128]  # Heatmap resolution
```

## 🐍 Using Python Client

Import and use the API client in your code:

```python
import requests

# Predict landmarks
with open("image.png", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)
    
result = response.json()
if result["success"]:
    for landmark in result["landmarks"]:
        print(f"{landmark['label']}: ({landmark['x']}, {landmark['y']})")
```

## 📦 Docker Usage (Optional)

Build and run in Docker:

```bash
docker build -t facial-landmark-api .
docker run -p 8000:8000 facial-landmark-api
```

## 🏆 Model Details

- **Architecture**: HRNet-W18 with custom head for 8-point landmarks
- **Input**: 512x512 RGB images
- **Output**: 8 landmark coordinates (x, y) in original image space
- **Preprocessing**: Affine crop + ImageNet normalization (matching training)

## ⚡ Performance Notes

- **GPU**: ~50-100 ms per image (NVIDIA GPU)
- **CPU**: ~200-500 ms per image
- **Memory**: ~2GB (GPU), ~1GB (CPU)

## 🔧 Troubleshooting

### Model not found error
Ensure `models/hrnet_finetuned_8pts.pth` exists in the working directory.

### CUDA out of memory
Run on CPU by modifying `Config.DEVICE` to `torch.device('cpu')`

### Image decoding error
Ensure the uploaded file is a valid JPEG or PNG image.

## 📝 License

See LICENSE file in the repository.
