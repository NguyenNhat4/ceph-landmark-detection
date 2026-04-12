"""
FastAPI endpoint for Facial Landmark Detection using HRNet.
Wraps the inference pipeline from inference.ipynb as a REST API.
"""

import os
import sys
import io
import torch
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import logging



from lib.utils.transforms import crop_v2
from lib.core.evaluation import decode_preds

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Initialize FastAPI app
# ============================================================================
app = FastAPI(
    title="Facial Landmark Detection API",
    description="API for detecting cephalometric facial landmarks using HRNet",
    version="1.0.0"
)

# ============================================================================
# Model Configuration
# ============================================================================
class Config:
    MODEL_PATH = "models/hrnet_finetuned_8pts.pth"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_KEYPOINTS = 8
    IMAGE_SIZE = [512, 512]
    HEATMAP_SIZE = [128, 128]
    
config_obj = Config()

# Landmark labels (8 cephalometric points)
LANDMARK_LABELS = [
    "Glabella",
    "N'",
    "Pronasal",
    "Subnasale",
    "Labiale sup",
    "Labiale inf",
    "B'",
    "Pog'"
]

# ============================================================================
# Model Definition
# ============================================================================
class PoseHRNet(torch.nn.Module):
    """HRNet model for facial landmark detection."""
    
    def __init__(self, num_keypoints=8):
        super(PoseHRNet, self).__init__()
        
        # Load hrnet_w18 from timm
        import timm
        import torch.nn as nn
        import torch.nn.functional as F
        
        self.backbone = timm.create_model('hrnet_w18', pretrained=False)
        out_channels_sum = 18 + 36 + 72 + 144
        
        self.final_head = nn.Sequential(
            nn.Conv2d(out_channels_sum, out_channels_sum, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels_sum),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels_sum, num_keypoints, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        import torch.nn.functional as F
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.conv2(x)
        x = self.backbone.bn2(x)
        x = self.backbone.act2(x)
        
        hr_features = self.backbone.stages(x)
        
        h, w = hr_features[0].shape[2:]
        out = [hr_features[0]]
        for i in range(1, len(hr_features)):
            out.append(F.interpolate(hr_features[i], size=(h, w), mode='bilinear', align_corners=False))
            
        out_cat = torch.cat(out, dim=1)
        heatmaps = self.final_head(out_cat)
        
        return heatmaps

# ============================================================================
# Global Model Instance
# ============================================================================
model = None

def load_model():
    """Load the trained HRNet model."""
    global model
    
    if model is not None:
        return model
    
    if not os.path.exists(config_obj.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {config_obj.MODEL_PATH}")
    
    logger.info(f"Loading model from {config_obj.MODEL_PATH}...")
    model = PoseHRNet(num_keypoints=config_obj.NUM_KEYPOINTS)
    
    state_dict = torch.load(config_obj.MODEL_PATH, map_location='cpu')
    
    try:
        model.load_state_dict(state_dict, strict=True)
        logger.info("✅ Model loaded successfully!")
    except RuntimeError as e:
        logger.warning(f"⚠️ Warning during load: {e}")
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(config_obj.DEVICE)
    model.eval()
    
    return model

# ============================================================================
# Inference Function
# ============================================================================
def predict_landmarks(image_array: np.ndarray, model) -> np.ndarray:
    """
    Predict facial landmarks on an image using HRNet.
    
    Args:
        image_array: Input image as numpy array (BGR format from cv2)
        model: Loaded PyTorch model
        
    Returns:
        Landmarks array of shape [num_joints, 2] with (x, y) coordinates
    """
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Get original image dimensions
    orig_h, orig_w = image_rgb.shape[:2]
    
    # Calculate center and scale
    center_w = orig_w / 2.0
    center_h = orig_h / 2.0
    center = np.array([center_w, center_h], dtype=np.float32)
    
    scale = max(orig_w, orig_h) / 200.0 * 1.25
    
    # Apply affine crop
    img_crop = crop_v2(image_rgb, center, scale, config_obj.IMAGE_SIZE, rot=0)
    
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    img_crop = img_crop.astype(np.float32)
    img_crop = (img_crop / 255.0 - mean) / std
    img_crop = img_crop.transpose([2, 0, 1])
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_crop).unsqueeze(0).float().to(config_obj.DEVICE)
    
    # Model inference
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
    
    # Decode predictions back to original image coordinates
    preds = decode_preds(
        output.cpu(),
        [torch.Tensor(center)],
        [scale],
        config_obj.HEATMAP_SIZE
    )
    
    # Return first batch
    return preds[0].numpy()

# ============================================================================
# Response Models
# ============================================================================
class LandmarkPoint(BaseModel):
    """Single landmark point with coordinates and label."""
    id: int
    label: str
    x: float
    y: float

class PredictionResponse(BaseModel):
    """API response with predicted landmarks."""
    success: bool
    landmarks: List[LandmarkPoint]
    message: str

# ============================================================================
# API Endpoints
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting up application...")
    try:
        load_model()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        raise

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": str(config_obj.DEVICE),
        "num_landmarks": config_obj.NUM_KEYPOINTS,
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Predict facial landmarks from an uploaded image.
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
        
    Returns:
        PredictionResponse with detected landmarks
    """
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Supported: JPEG, PNG"
        )
    
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        logger.info(f"Processing image: {file.filename} (size: {image.shape})")
        
        # Run inference
        landmarks = predict_landmarks(image, model)
        
        # Format response
        landmark_points = []
        for i, (x, y) in enumerate(landmarks):
            landmark_points.append(
                LandmarkPoint(
                    id=i,
                    label=LANDMARK_LABELS[i] if i < len(LANDMARK_LABELS) else f"Point_{i}",
                    x=float(x),
                    y=float(y)
                )
            )
        
        logger.info(f"Successfully predicted {len(landmark_points)} landmarks")
        
        return PredictionResponse(
            success=True,
            landmarks=landmark_points,
            message=f"Successfully detected {len(landmark_points)} facial landmarks"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/landmarks")
async def get_landmarks() -> Dict[str, List[str]]:
    """Get the list of supported landmarks."""
    return {
        "landmarks": LANDMARK_LABELS,
        "count": len(LANDMARK_LABELS)
    }

# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Facial Landmark Detection API...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
