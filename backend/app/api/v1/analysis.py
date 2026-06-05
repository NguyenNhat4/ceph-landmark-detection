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
from ...schemas.landmark import AnalysisResponse, AnalysisCreate
from ...db.models import Analysis, Image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])
image_service = ImageService()


@router.post("/save", response_model=AnalysisResponse, status_code=201)
async def save_analysis(
    analysis_data: AnalysisCreate,
    db: Session = Depends(get_db)
):
    """
    Save analysis results for a patient
    
    Args:
        analysis_data: Analysis creation schema (JSON)
    """
    try:
        # Verify patient exists
        patient = PatientService.get_patient(db, analysis_data.patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {analysis_data.patient_id} not found")

        # Verify image exists
        image = db.query(Image).filter(Image.id == analysis_data.image_id).first()
        if not image:
            raise HTTPException(status_code=404, detail=f"Image {analysis_data.image_id} not found")

        # Convert landmarks to List[dict] if provided
        landmarks_data = None
        if analysis_data.landmarks:
            landmarks_data = [lm.dict() for lm in analysis_data.landmarks]

        analysis = AnalysisService.create_analysis(
            db=db,
            patient_id=analysis_data.patient_id,
            image_id=analysis_data.image_id,
            landmarks=landmarks_data,
            confidence_score=analysis_data.confidence_score,
            status=analysis_data.status
        )

        if not analysis:
            raise HTTPException(status_code=400, detail="Failed to create analysis")

        return analysis

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{analysis_id}", response_model=AnalysisResponse)
async def update_analysis(
    analysis_id: int,
    analysis_data: AnalysisCreate,
    db: Session = Depends(get_db)
):
    """Update analysis results"""
    try:
        # Verify analysis exists
        analysis = AnalysisService.get_analysis(db, analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")

        # Convert landmarks to List[dict] if provided
        landmarks_data = None
        if analysis_data.landmarks:
            landmarks_data = [lm.dict() for lm in analysis_data.landmarks]

        updated_analysis = AnalysisService.update_analysis(
            db=db,
            analysis_id=analysis_id,
            landmarks=landmarks_data,
            confidence_score=analysis_data.confidence_score,
            status=analysis_data.status
        )

        if not updated_analysis:
            raise HTTPException(status_code=400, detail="Failed to update analysis")

        return updated_analysis

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating analysis: {e}", exc_info=True)
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
