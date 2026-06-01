"""Patient management API endpoints"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List

from ...db.database import get_db
from ...services.patient import PatientService
from ...schemas.landmark import PatientCreate, PatientUpdate, PatientResponse, ImageResponse
from ...db.models import Patient


router = APIRouter(prefix="/api/v1/patients", tags=["patients"])


@router.post("/", response_model=PatientResponse, status_code=201)
async def create_patient(
    patient_data: PatientCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new patient record
    
    Required fields:
    - **fullname**: Patient full name
    - **consultation_date**: Consultation date (ISO format: YYYY-MM-DDTHH:MM:SS)
    
    Optional fields:
    - **phone**: Patient phone number
    """
    try:
        patient = PatientService.create_patient(db, patient_data)
        return patient
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """Get patient information by ID"""
    patient = PatientService.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return patient


@router.get("/", response_model=List[PatientResponse])
async def list_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get all patients with pagination"""
    patients = PatientService.get_all_patients(db, skip=skip, limit=limit)
    return patients


@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(
    patient_id: int,
    patient_data: PatientUpdate,
    db: Session = Depends(get_db)
):
    """Update patient information"""
    patient = PatientService.update_patient(db, patient_id, patient_data)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return patient


@router.delete("/{patient_id}", status_code=204)
async def delete_patient(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """Delete a patient and all associated data"""
    success = PatientService.delete_patient(db, patient_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")


@router.get("/{patient_id}/images", response_model=List[ImageResponse])
async def get_patient_images(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """Get all images for a patient"""
    patient = PatientService.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    images = PatientService.get_patient_images(db, patient_id)
    return images


@router.get("/search", response_model=List[PatientResponse])
async def search_patients(
    q: str = Query(..., min_length=1, description="Search query (name or phone)"),
    db: Session = Depends(get_db)
):
    """Search patients by name or phone number"""
    patients = PatientService.search_patients(db, q)
    return patients
