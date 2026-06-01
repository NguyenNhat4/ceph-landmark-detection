"""Analysis results API endpoints"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
import json

from ...db.database import get_db
from ...services.patient import PatientService
from ...services.analysis import AnalysisService
from ...services.image import ImageService
from ...schemas.landmark import AnalysisResponse
from ...db.models import Analysis

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])
image_service = ImageService()


@router.post("/save", response_model=AnalysisResponse, status_code=201)
async def save_analysis(
    patient_id: int = Form(...),
    image_file: UploadFile = File(...),
    landmarks: Optional[str] = Form(None),
    confidence_score: Optional[float] = Form(None),
    notes: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Save analysis results for a patient
    
    Args:
        patient_id: ID of the patient
        image_file: Uploaded X-ray image
        landmarks: Detected landmarks (JSON string)
        confidence_score: Overall confidence score
        notes: Additional notes about the analysis
    """
    try:
        # Verify patient exists
        patient = PatientService.get_patient(db, patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

        # Create image record using ImageService
        db_image = image_service.create_image(
            db=db,
            patient_id=patient_id,
            file=image_file,
            image_type="xray"
        )
        
        if not db_image:
            raise HTTPException(status_code=400, detail="Failed to create image record")

        # Parse landmarks if provided
        landmarks_data = None
        if landmarks:
            try:
                landmarks_data = json.loads(landmarks)
            except json.JSONDecodeError:
                logger.warning("Invalid landmarks JSON format")

        # Create analysis record
        analysis = AnalysisService.create_analysis(
            db=db,
            patient_id=patient_id,
            image_id=db_image.id,
            landmarks=landmarks_data,
            confidence_score=confidence_score,
            notes=notes
        )

        if not analysis:
            raise HTTPException(status_code=400, detail="Failed to create analysis")

        return analysis

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """Get analysis by ID"""
    analysis = AnalysisService.get_analysis(db, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    return analysis


@router.get("/patient/{patient_id}", response_model=List[AnalysisResponse])
async def get_patient_analyses(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """Get all analyses for a patient"""
    patient = PatientService.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    analyses = PatientService.get_patient_analyses(db, patient_id)
    return analyses
