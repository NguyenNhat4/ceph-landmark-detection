from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List


# ============ Patient Schemas ============
class PatientBase(BaseModel):
    """Base patient schema"""
    fullname: str = Field(..., min_length=1, max_length=255, description="Patient full name")
    phone: Optional[str] = Field(None, max_length=20, description="Patient phone number")
    date_of_birth: Optional[datetime] = Field(None, description="Patient date of birth")
    chief_complaint: Optional[str] = Field(None, description="Patient chief complaint")
    consultation_date: datetime = Field(..., description="Consultation date")


class PatientCreate(PatientBase):
    """Schema for creating a new patient"""
    note: Optional[str] = Field(None, max_length=1000, description="Optional initial note for the patient")


class PatientUpdate(BaseModel):
    """Schema for updating patient information"""
    fullname: Optional[str] = Field(None, max_length=255)
    phone: Optional[str] = Field(None, max_length=20)
    date_of_birth: Optional[datetime] = None
    chief_complaint: Optional[str] = None
    consultation_date: Optional[datetime] = None
    note: Optional[str] = Field(None, max_length=1000, description="Add a new note for the patient")


class NoteResponse(BaseModel):
    """Schema for note response"""
    id: int
    content: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class PatientResponse(PatientBase):
    """Schema for patient response"""
    id: int
    created_at: datetime
    updated_at: datetime
    notes: List[NoteResponse] = []

    class Config:
        from_attributes = True


# ============ Landmark/Coordinate Schemas ============
class Coordinate(BaseModel):
    """Schema for a single coordinate point (x, y)"""
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence score")

    class Config:
        from_attributes = True


class Landmark(BaseModel):
    """Schema for a landmark with coordinates"""
    name: str = Field(..., description="Landmark name")
    coordinate: Coordinate
    id: Optional[int] = None


# ============ Analysis Schemas ============
class AnalysisBase(BaseModel):
    """Base analysis schema"""
    landmarks: Optional[List[Landmark]] = None
    confidence_score: Optional[float] = Field(None, ge=0, le=1)
    status: Optional[str] = None


class AnalysisCreate(AnalysisBase):
    """Schema for creating analysis"""
    patient_id: int
    image_id: int


class AnalysisResponse(AnalysisBase):
    """Schema for analysis response"""
    id: int
    patient_id: int
    image_id: int
    analysis_date: datetime

    class Config:
        from_attributes = True


# ============ Image Schemas ============
class ImageResponse(BaseModel):
    """Schema for image response"""
    id: int
    patient_id: int
    filename: str
    image_type: Optional[str]
    upload_date: datetime
    image_url: Optional[str] = None

    class Config:
        from_attributes = True

class PatientImagesResponse(BaseModel):
    """Schema for specific patient images (xray, frontal, profile)"""
    xray: Optional[ImageResponse] = None
    frontal: Optional[ImageResponse] = None
    profile: Optional[ImageResponse] = None
