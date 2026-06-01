"""Image management API endpoints"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.orm import Session
from typing import List

from ...db.database import get_db
from ...services.image import ImageService
from ...services.patient import PatientService
from ...schemas.landmark import ImageResponse

router = APIRouter(prefix="/api/v1/images", tags=["images"])
image_service = ImageService()


@router.post("/upload", response_model=ImageResponse, status_code=201)
async def upload_image(
    patient_id: int = Query(...),
    file: UploadFile = File(...),
    image_type: str = Query("xray"),
    db: Session = Depends(get_db)
):
    """
    Upload an X-ray image for a patient
    
    Args:
        patient_id: Patient ID
        file: Image file to upload
        image_type: Type of image (lateral, panoramic, xray, etc.)
    """
    try:
        # Verify patient exists
        patient = PatientService.get_patient(db, patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

        # Create image record
        db_image = image_service.create_image(
            db=db,
            patient_id=patient_id,
            file=file,
            image_type=image_type
        )

        if not db_image:
            raise HTTPException(status_code=400, detail="Failed to upload image")

        return db_image

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{image_id}", response_model=ImageResponse)
async def get_image(
    image_id: int,
    db: Session = Depends(get_db)
):
    """Get image information by ID"""
    image = image_service.get_image(db, image_id)
    if not image:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
    return image


@router.get("/patient/{patient_id}", response_model=List[ImageResponse])
async def get_patient_images(
    patient_id: int,
    image_type: str = Query(None),
    db: Session = Depends(get_db)
):
    """
    Get all images for a patient
    
    Args:
        patient_id: Patient ID
        image_type: Optional filter by image type (lateral, panoramic, xray, etc.)
    """
    # Verify patient exists
    patient = PatientService.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    if image_type:
        images = image_service.get_patient_images_by_type(db, patient_id, image_type)
    else:
        images = image_service.get_patient_images(db, patient_id)
    
    return images


@router.delete("/{image_id}", status_code=204)
async def delete_image(
    image_id: int,
    db: Session = Depends(get_db)
):
    """Delete an image and its associated file"""
    success = image_service.delete_image(db, image_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")


@router.get("/patient/{patient_id}/latest", response_model=ImageResponse)
async def get_latest_patient_image(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """Get the most recently uploaded image for a patient"""
    # Verify patient exists
    patient = PatientService.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    image = image_service.get_latest_patient_image(db, patient_id)
    if not image:
        raise HTTPException(status_code=404, detail="No images found for this patient")
    
    return image
