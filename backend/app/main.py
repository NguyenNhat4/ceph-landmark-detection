import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.lifespan import lifespan
from .db.database import init_db
from .api.v1.inference import router as inference_router
from .api.v1.patients import router as patients_router
from .api.v1.analysis import router as analysis_router
from .api.v1.images import router as images_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database on startup
try:
    init_db()
    logger.info("Database initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")

# Create FastAPI app with lifespan management
app = FastAPI(
    title="Cephalometric Landmark Detection API",
    description="HRNet-based API for detecting cephalometric landmarks in X-ray images",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
# In backend/app/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], # <-- Change this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(inference_router)
app.include_router(patients_router)
app.include_router(analysis_router)
app.include_router(images_router)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cephalometric Landmark Detection API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/api/health",
            "setup": "/api/setup",
            "predict": "/api/predict"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
