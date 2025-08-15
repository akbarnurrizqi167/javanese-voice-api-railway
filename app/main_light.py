"""
Ultra-light version for Railway deployment testing
"""
import os
import logging
from datetime import datetime
from typing import Optional
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Form
    from fastapi.middleware.cors import CORSMiddleware
    from pydub import AudioSegment
    import io
    logger.info("✅ Basic dependencies loaded")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    raise e

# Initialize FastAPI
app = FastAPI(
    title="Javanese Voice API - Light",
    description="Light version for Railway deployment testing",
    version="2.0.0-light",
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

start_time = datetime.now()
upload_count = 0

CLASSES = [
    'ha', 'na', 'ca', 'ra', 'ka', 'da', 'ta', 'sa', 'wa', 'la',
    'pa', 'dha', 'ja', 'ya', 'nya', 'ma', 'ga', 'ba', 'tha', 'nga'
]

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Javanese Voice API - Light Version",
        "version": "2.0.0-light",
        "status": "running",
        "platform": "Railway",
        "mode": "testing",
        "ml_loaded": False,
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global start_time, upload_count
    uptime = (datetime.now() - start_time).total_seconds()
    
    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "uploads_processed": upload_count,
        "platform": "Railway",
        "version": "light",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/supported-aksara")
async def supported_aksara():
    """Get list of supported Javanese aksara"""
    return {
        "count": len(CLASSES),
        "aksara": CLASSES,
        "note": "ML prediction not available in light version"
    }

@app.post("/test-upload")
async def test_upload(file: UploadFile = File(...)):
    """Test file upload without prediction"""
    global upload_count
    
    try:
        content = await file.read()
        file_size = len(content)
        
        # Basic audio format check
        audio_info = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": file_size,
            "size_kb": round(file_size / 1024, 2)
        }
        
        # Try basic audio processing if possible
        try:
            audio = AudioSegment.from_file(io.BytesIO(content))
            audio_info.update({
                "duration_ms": len(audio),
                "channels": audio.channels,
                "frame_rate": audio.frame_rate,
                "sample_width": audio.sample_width
            })
        except Exception as e:
            audio_info["audio_processing_error"] = str(e)
        
        upload_count += 1
        
        return {
            "success": True,
            "file_info": audio_info,
            "uploads_processed": upload_count,
            "message": "File upload test successful",
            "note": "ML prediction not available in light version"
        }
        
    except Exception as e:
        logger.error(f"Upload test error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload test failed: {str(e)}")

@app.post("/mock-predict")
async def mock_predict(
    file: UploadFile = File(...),
    target: Optional[str] = Form(None)
):
    """Mock prediction for testing"""
    global upload_count
    
    try:
        content = await file.read()
        
        # Mock prediction
        import random
        random.seed(42)  # For consistent testing
        
        predicted_class = random.choice(CLASSES)
        confidence = round(random.uniform(0.7, 0.95), 3)
        
        upload_count += 1
        
        return {
            "success": True,
            "prediction": predicted_class,
            "confidence": confidence,
            "target": target,
            "is_correct": predicted_class == target if target else None,
            "file_info": {
                "size_kb": round(len(content) / 1024, 2),
                "filename": file.filename
            },
            "note": "This is a MOCK prediction for testing only",
            "version": "light"
        }
        
    except Exception as e:
        logger.error(f"Mock prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Mock prediction failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    global start_time, upload_count
    uptime = (datetime.now() - start_time).total_seconds()
    
    return {
        "total_uploads": upload_count,
        "uptime_seconds": uptime,
        "uptime_formatted": f"{uptime//3600:.0f}h {(uptime%3600)//60:.0f}m",
        "start_time": start_time.isoformat(),
        "version": "light",
        "platform": "Railway",
        "supported_classes": len(CLASSES)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting light server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
