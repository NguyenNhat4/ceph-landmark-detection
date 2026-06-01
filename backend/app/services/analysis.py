"""Analysis service for storing landmark detection results"""
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from ..db.models import Analysis, Image, Patient
from ..schemas.landmark import AnalysisCreate, Landmark


class AnalysisService:
    """Service for managing analysis records"""

    @staticmethod
    def create_analysis(
        db: Session, 
        patient_id: int,
        image_id: int,
        landmarks: Optional[List[dict]] = None,
        confidence_score: Optional[float] = None,
        notes: Optional[str] = None
    ) -> Optional[Analysis]:
        """
        Create a new analysis record after landmark detection
        
        Args:
            db: Database session
            patient_id: Patient ID
            image_id: Image ID
            landmarks: Detected landmarks data
            confidence_score: Overall confidence score
            notes: Additional notes
            
        Returns:
            Created analysis object
        """
        # Verify patient and image exist
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        image = db.query(Image).filter(Image.id == image_id).first()
        
        if not patient or not image:
            return None

        db_analysis = Analysis(
            patient_id=patient_id,
            image_id=image_id,
            landmarks=landmarks,
            confidence_score=confidence_score,
            notes=notes,
            analysis_date=datetime.utcnow()
        )
        db.add(db_analysis)
        db.commit()
        db.refresh(db_analysis)
        return db_analysis

    @staticmethod
    def get_analysis(db: Session, analysis_id: int) -> Optional[Analysis]:
        """Get analysis by ID"""
        return db.query(Analysis).filter(Analysis.id == analysis_id).first()

    @staticmethod
    def get_patient_latest_analysis(db: Session, patient_id: int) -> Optional[Analysis]:
        """Get latest analysis for a patient"""
        return db.query(Analysis).filter(
            Analysis.patient_id == patient_id
        ).order_by(Analysis.analysis_date.desc()).first()

    @staticmethod
    def get_image_analyses(db: Session, image_id: int) -> List[Analysis]:
        """Get all analyses for an image"""
        return db.query(Analysis).filter(Analysis.image_id == image_id).all()

