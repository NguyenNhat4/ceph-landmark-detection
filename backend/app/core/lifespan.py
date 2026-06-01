import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from ..ml.landmark_detector import load_model

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage FastAPI application lifecycle:
    - Load model on startup
    - Cleanup on shutdown
    """
    # Startup
    logger.info("Application startup - Loading model...")
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
        # Continue running even if model fails to load
    
    yield
    
    # Shutdown
    logger.info("Application shutdown")

