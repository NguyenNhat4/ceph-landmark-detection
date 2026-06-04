"""Patient service for managing patient data"""
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Optional
from datetime import datetime

from ..db.models import Patient, Image, Analysis
from ..schemas.landmark import PatientCreate, PatientUpdate, PatientResponse


class PatientService:
    """Service for patient management operations"""

    @staticmethod
    def create_patient(db: Session, patient_data: PatientCreate) -> Patient:
        """
        Create a new patient record
        
        Args:
            db: Database session
            patient_data: Patient creation data
            
        Returns:
            Created patient object
            
        Raises:
            IntegrityError: If phone number already exists
        """
        try:
            db_patient = Patient(
                fullname=patient_data.fullname,
                phone=patient_data.phone,
                consultation_date=patient_data.consultation_date
            )
            db.add(db_patient)
            db.commit()
            db.refresh(db_patient)
            
            if patient_data.note:
                PatientService.create_patient_note(db, db_patient.id, patient_data.note)
                db.refresh(db_patient)
                
            return db_patient
        except IntegrityError as e:
            db.rollback()
            if "phone" in str(e):
                raise ValueError(f"Phone number {patient_data.phone} already exists")
            raise

    @staticmethod
    def create_patient_note(db: Session, patient_id: int, content: str) -> 'Note':
        """Create a new note for a patient"""
        from ..db.models import Note
        db_note = Note(patient_id=patient_id, content=content)
        db.add(db_note)
        db.commit()
        db.refresh(db_note)
        return db_note

    @staticmethod
    def get_patient(db: Session, patient_id: int) -> Optional[Patient]:
        """Get patient by ID"""
        return db.query(Patient).filter(Patient.id == patient_id).first()

    @staticmethod
    def get_patient_by_phone(db: Session, phone: str) -> Optional[Patient]:
        """Get patient by phone number"""
        return db.query(Patient).filter(Patient.phone == phone).first()

    @staticmethod
    def get_all_patients(db: Session, skip: int = 0, limit: int = 100) -> List[Patient]:
        """Get all patients with pagination"""
        return db.query(Patient).offset(skip).limit(limit).all()

    @staticmethod
    def update_patient(db: Session, patient_id: int, patient_data: PatientUpdate) -> Optional[Patient]:
        """Update patient information"""
        db_patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not db_patient:
            return None

        update_data = patient_data.dict(exclude_unset=True)
        note_content = update_data.pop("note", None)
        
        for field, value in update_data.items():
            setattr(db_patient, field, value)
        
        db_patient.updated_at = datetime.utcnow()
        
        if note_content:
            PatientService.create_patient_note(db, patient_id, note_content)
            
        db.commit()
        db.refresh(db_patient)
        return db_patient

    @staticmethod
    def delete_patient(db: Session, patient_id: int) -> bool:
        """Delete a patient and all associated data"""
        db_patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not db_patient:
            return False
        
        db.delete(db_patient)
        db.commit()
        return True

    @staticmethod
    def get_patient_analyses(db: Session, patient_id: int) -> List[Analysis]:
        """Get all analyses for a patient"""
        return db.query(Analysis).filter(Analysis.patient_id == patient_id).all()

    @staticmethod
    def get_patient_images(db: Session, patient_id: int) -> List[Image]:
        """Get all images for a patient"""
        return db.query(Image).filter(Image.patient_id == patient_id).all()

    @staticmethod
    def search_patients(db: Session, query: str) -> List[Patient]:
        """Search patients by name or phone"""
        return db.query(Patient).filter(
            (Patient.fullname.ilike(f"%{query}%")) | 
            (Patient.phone.ilike(f"%{query}%"))
        ).all()
