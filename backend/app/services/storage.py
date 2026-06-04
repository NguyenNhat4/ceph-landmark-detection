"""Storage service for handling file operations"""
import os
import uuid
from pathlib import Path
from fastapi import UploadFile
import logging

logger = logging.getLogger(__name__)


class StorageService:
    """Service for managing file storage"""
    
    def __init__(self):
        # Create storage directory if it doesn't exist
        self.storage_dir = os.getenv("STORAGE_DIR", "/tmp/cephalometric_storage")
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)
        self.images_dir = os.path.join(self.storage_dir, "images")
        Path(self.images_dir).mkdir(parents=True, exist_ok=True)

    def save_image(self, file: UploadFile, patient_id: int) -> str:
        """
        Save uploaded image to storage
        
        Args:
            file: Uploaded file
            patient_id: Patient ID for organizing files
            
        Returns:
            File path relative to storage directory
        """
        try:
            # Create patient directory
            patient_dir = os.path.join(self.images_dir, f"patient_{patient_id}")
            Path(patient_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            file_ext = os.path.splitext(file.filename)[1]
            unique_filename = f"{uuid.uuid4()}{file_ext}"
            file_path = os.path.join(patient_dir, unique_filename)
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(file.file.read())
            
            logger.info(f"Image saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            raise

    def get_image_path(self, file_path: str) -> str:
        """Get full file path for a stored image"""
        return os.path.join(self.storage_dir, file_path)

    def delete_image(self, file_path: str) -> bool:
        """Delete an image file"""
        try:
            full_path = self.get_image_path(file_path)
            if os.path.exists(full_path):
                os.remove(full_path)
                logger.info(f"Image deleted: {full_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting image: {e}")
            return False
