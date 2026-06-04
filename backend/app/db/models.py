from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base


class Patient(Base):
    """Patient model for storing patient information"""
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    fullname = Column(String(255), nullable=False, index=True)
    phone = Column(String(20), nullable=True, unique=True, index=True)
    consultation_date = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    images = relationship("Image", back_populates="patient", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="patient", cascade="all, delete-orphan")
    notes = relationship("Note", back_populates="patient", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Patient(id={self.id}, fullname={self.fullname}, phone={self.phone})>"


class Image(Base):
    """Image model for storing patient X-ray images"""
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    image_type = Column(String(50), nullable=True)  # e.g., "lateral", "panoramic"
    upload_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="images")
    analyses = relationship("Analysis", back_populates="image", cascade="all, delete-orphan")

    @property
    def image_url(self) -> str:
        """Returns the relative URL to access the image file"""
        return f"/api/v1/images/{self.id}/file"


class Analysis(Base):
    """Analysis model for storing landmark detection results"""
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False, index=True)
    landmarks = Column(JSON, nullable=True)  # Store landmarks as JSON
    confidence_score = Column(Float, nullable=True)
    analysis_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="analyses")
    image = relationship("Image", back_populates="analyses")


class Note(Base):
    """Note model for storing notes, to be used for semantic search"""
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    content = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="notes")
