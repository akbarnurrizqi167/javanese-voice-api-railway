"""
Javanese Aksara Voice Recognition API - Railway Minimal Version
FastAPI backend without heavy dependencies for initial deployment
"""

import os
import logging
from datetime import datetime
from typing import Optional, List
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Form
    from fastapi.middleware.cors import CORSMiddleware
    logger.info("✅ Core dependencies loaded successfully")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    raise e

# Initialize FastAPI
app = FastAPI(
    title="Javanese Aksara Voice Recognition API",
    description="Minimal API for Railway deployment testing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
start_time = datetime.now()
prediction_count = 0

# Javanese aksara classes
CLASSES = [
    'ha', 'na', 'ca', 'ra', 'ka', 'da', 'ta', 'sa', 'wa', 'la',
    'pa', 'dha', 'ja', 'ya', 'nya', 'ma', 'ga', 'ba', 'tha', 'nga'
]

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info("🚀 Starting Javanese Aksara Recognition API (Minimal Version)...")
    logger.info("✅ API ready for testing")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Javanese Aksara Voice Recognition API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs",
        "platform": "Railway",
        "deployment": "minimal"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global start_time, prediction_count
    
    uptime = (datetime.now() - start_time).total_seconds()
    
    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "predictions_made": prediction_count,
        "timestamp": datetime.now().isoformat(),
        "deployment": "minimal"
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_name": "Javanese Aksara CNN (Minimal Deployment)",
        "version": "1.0.0",
        "status": "deployment_test",
        "classes": len(CLASSES),
        "supported_aksara": CLASSES,
        "note": "Full model loading requires additional dependencies"
    }

@app.get("/supported-aksara")
async def supported_aksara():
    """Get list of supported Javanese aksara"""
    return {
        "count": len(CLASSES),
        "aksara": CLASSES,
        "description": "20 traditional Javanese aksara characters"
    }

@app.post("/predict")
async def predict_minimal(
    file: UploadFile = File(...),
    target: Optional[str] = Form(None)
):
    """Minimal prediction endpoint for testing"""
    global prediction_count
    
    try:
        # Read file info
        content = await file.read()
        file_size = len(content)
        
        prediction_count += 1
        
        # Return mock prediction for testing
        result = {
            "success": True,
            "prediction": "ha",  # Mock prediction
            "confidence": 0.85,
            "target": target,
            "is_correct": target == "ha" if target else None,
            "file_info": {
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": file_size
            },
            "metadata": {
                "model_version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
                "note": "Mock prediction - full model requires additional dependencies"
            }
        }
        
        logger.info(f"Mock prediction: ha (file: {file.filename}, {file_size} bytes)")
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    global start_time, prediction_count
    
    uptime = (datetime.now() - start_time).total_seconds()
    
    return {
        "total_predictions": prediction_count,
        "uptime_seconds": uptime,
        "uptime_formatted": f"{uptime//3600:.0f}h {(uptime%3600)//60:.0f}m",
        "start_time": start_time.isoformat(),
        "supported_classes": len(CLASSES),
        "platform": "Railway",
        "deployment": "minimal"
    }

@app.post("/test-upload")
async def test_upload(file: UploadFile = File(...)):
    """Test file upload functionality"""
    try:
        content = await file.read()
        return {
            "success": True,
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(content),
            "size_kb": round(len(content) / 1024, 2),
            "status": "upload_successful"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload test failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
