from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import cv2
import numpy as np
from model import LandmarkModel
import io
from typing import List
import uvicorn

app = FastAPI(title="Cephalometric Landmarks Detection API")

# CORS middleware để frontend có thể gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origins, có thể cấu hình cụ thể hơn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tên các landmarks (29 điểm) - khớp với dữ liệu training
LANDMARK_NAMES = [
    "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R", "S",
    "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
    "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N'", "Pog'", "Sn"
]

# Tên đầy đủ của các landmarks
LANDMARK_FULL_NAMES = {
    "A": "A-point",
    "ANS": "Anterior Nasal Spine",
    "B": "B-point",
    "Me": "Menton",
    "N": "Nasion",
    "Or": "Orbitale",
    "Pog": "Pogonion",
    "PNS": "Posterior Nasal Spine",
    "Pn": "Pronasale",
    "R": "Ramus",
    "S": "Sella",
    "Ar": "Articulare",
    "Co": "Condylion",
    "Gn": "Gnathion",
    "Go": "Gonion",
    "Po": "Porion",
    "LPM": "Lower 2nd PM Cusp Tip",
    "LIT": "Lower Incisor Tip",
    "LMT": "Lower Molar Cusp Tip",
    "UPM": "Upper 2nd PM Cusp Tip",
    "UIA": "Upper Incisor Apex",
    "UIT": "Upper Incisor Tip",
    "UMT": "Upper Molar Cusp Tip",
    "LIA": "Lower Incisor Apex",
    "Li": "Labrale inferius",
    "Ls": "Labrale superius",
    "N'": "Soft Tissue Nasion",
    "Pog'": "Soft Tissue Pogonion",
    "Sn": "Subnasale"
}

# Global model variable
model = None
device = None

def load_model():
    """Load trained model at startup"""
    global model, device

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model on device: {device}")

    model = LandmarkModel(num_landmarks=29, backbone='efficientnet_b3')

    # Load checkpoint
    checkpoint_path = 'checkpoints_full/checkpoint_epoch_100.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully from {checkpoint_path}")
    print(f"  Trained epoch: {checkpoint['epoch']}")
    print(f"  Val MRE: {checkpoint['val_mre']:.2f}mm")

@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    load_model()

def preprocess_image(image_bytes: bytes, img_size=512):
    """Preprocess image for model input"""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Cannot decode image")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # Resize
    img_resized = cv2.resize(image, (img_size, img_size))

    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = (img_normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    # To tensor
    img_tensor = torch.FloatTensor(img_normalized).permute(2, 0, 1).unsqueeze(0)

    return img_tensor, (h, w)

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "running",
        "message": "Cephalometric Landmarks Detection API",
        "version": "1.0.0",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict_landmarks(file: UploadFile = File(...)):
    """
    Upload ảnh X-ray và nhận về JSON chứa tọa độ các landmarks

    Returns:
        JSON với format tương tự cks2ip8fq2a0j0yufdfssbc09.json
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Kiểm tra file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Đọc file
        contents = await file.read()

        # Preprocess
        img_tensor, (h, w) = preprocess_image(contents)

        # Predict
        with torch.no_grad():
            pred = model(img_tensor.to(device))

        # Denormalize landmarks (chuyển về tọa độ ảnh gốc)
        landmarks = pred[0].cpu().numpy() * [w, h]

        # Tạo JSON response theo format của frontend
        result = {
            "ceph_id": "api_generated",
            "landmarks": []
        }

        for i, (x, y) in enumerate(landmarks):
            symbol = LANDMARK_NAMES[i]
            result["landmarks"].append({
                "landmark_id": f"api_landmark_{i}",
                "title": LANDMARK_FULL_NAMES.get(symbol, symbol),
                "symbol": symbol,
                "value": {
                    "x": int(round(x)),
                    "y": int(round(y))
                }
            })

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict-simple")
async def predict_landmarks_simple(file: UploadFile = File(...)):
    """
    Upload ảnh X-ray và nhận về JSON đơn giản hơn (chỉ coordinates)

    Returns:
        JSON với format đơn giản: {"landmarks": [{"symbol": "A", "x": 100, "y": 200}, ...]}
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        img_tensor, (h, w) = preprocess_image(contents)

        with torch.no_grad():
            pred = model(img_tensor.to(device))

        landmarks = pred[0].cpu().numpy() * [w, h]

        result = {
            "landmarks": [
                {
                    "symbol": LANDMARK_NAMES[i],
                    "x": float(x),
                    "y": float(y)
                }
                for i, (x, y) in enumerate(landmarks)
            ]
        }

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    # Chạy server tại port 8006
    uvicorn.run(app, host="0.0.0.0", port=8006)
