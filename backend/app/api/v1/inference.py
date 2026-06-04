import logging
import torch
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
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


from pydantic import BaseModel
from sqlalchemy.orm import Session
import os
from ...db.database import get_db
from ...services.image import ImageService
from ...services.analysis import AnalysisService

image_service = ImageService()

class PredictRequest(BaseModel):
    image_id: int
    image_type: str

@router.post("/predict")
async def predict(
    request: PredictRequest,
    db: Session = Depends(get_db)
):
    """
    API /predict - Predict landmarks from uploaded image and save analysis
    
    Args:
        request: PredictRequest with image_id and image_type
        
    Returns:
        Landmarks with coordinates and confidence scores
    """
    try:
        if request.image_type != "xray":
            return JSONResponse(status_code=400, content={"error": "Prediction is currently only supported for xray images"})

        image_record = image_service.get_image(db, request.image_id)
        if not image_record:
            return JSONResponse(status_code=404, content={"error": "Image not found"})

        file_path = image_service.storage.get_image_path(image_record.file_path)
        if not os.path.exists(file_path):
            return JSONResponse(status_code=404, content={"error": "Image file not found on disk"})

        with open(file_path, "rb") as f:
            image_bytes = f.read()

        image = _decode_image(image_bytes)

        landmarks, width, height, roi_bbox = predict_landmarks(image)

        # Save analysis
        AnalysisService.create_analysis(
            db=db,
            patient_id=image_record.patient_id,
            image_id=request.image_id,
            landmarks=landmarks,
            confidence_score=None
        )

        response = {
            "landmarks": landmarks
        }

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
