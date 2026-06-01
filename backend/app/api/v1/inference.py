import logging
import torch
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from ...ml.landmark_detector import (
    predict_landmarks, _decode_image, MODEL_LOADED,
    NUM_JOINTS, LANDMARK_SYMBOLS, IMAGE_SIZE, HEATMAP_SIZE
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["inference"])


@router.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "service": "Cephalometric Landmark Detection API"
    }


@router.post("/setup")
async def setup():
    """Setup/configuration endpoint - returns model info and API metadata"""
    return {
        "model_version": "HRNet-W32 (ImageNet Pretrained)",
        "status": "ready" if MODEL_LOADED else "initializing",
        "num_landmarks": NUM_JOINTS,
        "landmark_symbols": LANDMARK_SYMBOLS,
        "input_size": IMAGE_SIZE,
        "heatmap_size": HEATMAP_SIZE,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "service": "Cephalometric Landmark Detection API v1.0",
        "endpoints": {
            "health": "/api/health",
            "predict": "/api/predict",
            "setup": "/api/setup",
            "docs": "/docs"
        }
    }


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    API /predict - Predict landmarks from uploaded image
    
    Args:
        file: Image file (JPG, PNG, etc.)
        
    Returns:
        Landmarks with coordinates and confidence scores
    """
    try:
        image_bytes = await file.read()
        image = _decode_image(image_bytes)

        landmarks, width, height, roi_bbox = predict_landmarks(image)

        # Format response to match frontend expectations
        response = {
            "landmarks": landmarks
        }

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
