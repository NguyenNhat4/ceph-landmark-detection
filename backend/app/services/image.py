"""Image service for managing patient X-ray images"""
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from ..db.models import Image, Patient
from ..services.storage import StorageService
from fastapi import UploadFile


class ImageService:
    """Service for managing image records and storage"""

    def __init__(self, storage: Optional[StorageService] = None):
        self.storage = storage or StorageService()

    def create_image(
        self,
        db: Session,
        patient_id: int,
        file: UploadFile,
        image_type: str = "xray"
    ) -> Optional[Image]:
        """
        Create a new image record and save the file
        
        Args:
            db: Database session
            patient_id: Patient ID
            file: Uploaded image file
            image_type: Type of image (lateral, panoramic, xray, etc.)
            
        Returns:
            Created image object or None if patient doesn't exist
        """
        # Verify patient exists
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            return None

        # Save file to storage
        file_path = self.storage.save_image(file, patient_id)
        
        # Create image record in database
        db_image = Image(
            patient_id=patient_id,
            filename=file.filename,
            file_path=file_path,
            image_type=image_type,
            upload_date=datetime.utcnow()
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)
        return db_image

    def get_image(self, db: Session, image_id: int) -> Optional[Image]:
        """Get image by ID"""
        return db.query(Image).filter(Image.id == image_id).first()

    def get_patient_images(self, db: Session, patient_id: int) -> List[Image]:
        """Get all images for a patient"""
        return db.query(Image).filter(Image.patient_id == patient_id).all()

    def get_image_by_filename(self, db: Session, filename: str) -> Optional[Image]:
        """Get image by filename"""
        return db.query(Image).filter(Image.filename == filename).first()

    def delete_image(self, db: Session, image_id: int) -> bool:
        """Delete an image and its file"""
        db_image = db.query(Image).filter(Image.id == image_id).first()
        if not db_image:
            return False

        # Delete file from storage
        self.storage.delete_image(db_image.file_path)
        
        # Delete database record
        db.delete(db_image)
        db.commit()
        return True

    def update_image_type(
        self,
        db: Session,
        image_id: int,
        image_type: str
    ) -> Optional[Image]:
        """Update image type"""
        db_image = db.query(Image).filter(Image.id == image_id).first()
        if not db_image:
            return None

        db_image.image_type = image_type
        db.commit()
        db.refresh(db_image)
        return db_image

    def get_latest_patient_image(self, db: Session, patient_id: int) -> Optional[Image]:
        """Get the most recently uploaded image for a patient"""
        return db.query(Image).filter(
            Image.patient_id == patient_id
        ).order_by(Image.upload_date.desc()).first()

    def get_patient_images_by_type(
        self,
        db: Session,
        patient_id: int,
        image_type: str
    ) -> List[Image]:
        """Get images for a patient filtered by type"""
        return db.query(Image).filter(
            Image.patient_id == patient_id,
            Image.image_type == image_type
        ).all()
