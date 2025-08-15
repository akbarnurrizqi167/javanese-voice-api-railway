"""
Javanese Aksara Voice Recognition API - Production Version with Real ML Model
FastAPI backend with full ML model integration for Railway deployment
"""

import os
import logging
import pickle
import numpy as np
import librosa
from datetime import datetime
from typing import Optional, List
import uvicorn
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Form
    from fastapi.middleware.cors import CORSMiddleware
    import tensorflow as tf
    from pydub import AudioSegment
    import soundfile as sf
    logger.info("✅ All dependencies loaded successfully")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    raise e

# Initialize FastAPI
app = FastAPI(
    title="Javanese Aksara Voice Recognition API",
    description="Production API with real ML model for Javanese voice recognition",
    version="2.0.0",
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
model = None
label_encoder = None

# Javanese aksara classes
CLASSES = [
    'ha', 'na', 'ca', 'ra', 'ka', 'da', 'ta', 'sa', 'wa', 'la',
    'pa', 'dha', 'ja', 'ya', 'nya', 'ma', 'ga', 'ba', 'tha', 'nga'
]

def load_model():
    """Load the trained model and label encoder"""
    global model, label_encoder
    
    try:
        # Load model
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "javanese_enhanced_retrain.h5")
        model = tf.keras.models.load_model(model_path)
        logger.info(f"✅ Model loaded from {model_path}")
        
        # Load label encoder
        encoder_path = os.path.join(os.path.dirname(__file__), "..", "models", "label_encoder_retrain.pkl")
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        logger.info(f"✅ Label encoder loaded from {encoder_path}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

def extract_features(audio_data, sr=22050, n_mfcc=13, max_len=100):
    """Extract MFCC features from audio data"""
    try:
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        
        # Pad or truncate to fixed length
        if mfcc.shape[1] < max_len:
            # Pad with zeros
            mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
        else:
            # Truncate
            mfcc = mfcc[:, :max_len]
        
        return mfcc.T  # Transpose to (time_steps, features)
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        raise e

def preprocess_audio(audio_file_content):
    """Preprocess audio file for prediction"""
    try:
        # Load audio using pydub first for format compatibility
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_file_content))
        
        # Convert to mono and set sample rate
        audio_segment = audio_segment.set_channels(1).set_frame_rate(22050)
        
        # Export to wav format in memory
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format='wav')
        wav_io.seek(0)
        
        # Load with librosa for feature extraction
        audio_data, sr = librosa.load(wav_io, sr=22050)
        
        # Extract features
        features = extract_features(audio_data, sr)
        
        # Reshape for model input (batch_size, time_steps, features)
        features = features.reshape(1, features.shape[0], features.shape[1])
        
        return features, len(audio_data), sr
        
    except Exception as e:
        logger.error(f"Audio preprocessing error: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Startup event - load model"""
    logger.info("🚀 Starting Javanese Aksara Recognition API (Production Version)...")
    
    model_loaded = load_model()
    if model_loaded:
        logger.info("✅ API ready with ML model loaded")
    else:
        logger.warning("⚠️ API started but model loading failed")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Javanese Aksara Voice Recognition API",
        "version": "2.0.0",
        "status": "active",
        "docs": "/docs",
        "platform": "Railway",
        "deployment": "production",
        "model_loaded": model is not None,
        "label_encoder_loaded": label_encoder is not None
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
        "deployment": "production",
        "model_status": "loaded" if model is not None else "not_loaded",
        "encoder_status": "loaded" if label_encoder is not None else "not_loaded"
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    model_status = "loaded" if model is not None else "not_loaded"
    
    info = {
        "model_name": "Javanese Aksara CNN Enhanced Retrain",
        "version": "2.0.0",
        "status": model_status,
        "classes": len(CLASSES),
        "supported_aksara": CLASSES,
        "input_shape": model.input_shape if model else None,
        "output_shape": model.output_shape if model else None
    }
    
    if model:
        info["model_summary"] = {
            "total_params": model.count_params(),
            "trainable_params": model.count_params() if hasattr(model, 'count_params') else 'unknown'
        }
    
    return info

@app.get("/supported-aksara")
async def supported_aksara():
    """Get list of supported Javanese aksara"""
    return {
        "count": len(CLASSES),
        "aksara": CLASSES,
        "description": "20 traditional Javanese aksara characters",
        "model_ready": model is not None
    }

@app.post("/predict")
async def predict_voice(
    file: UploadFile = File(...),
    target: Optional[str] = Form(None)
):
    """Real prediction endpoint using ML model"""
    global prediction_count
    
    if model is None or label_encoder is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('audio/'):
            if not file.filename or not any(file.filename.lower().endswith(ext) for ext in ['.wav', '.mp3', '.m4a', '.ogg']):
                raise HTTPException(
                    status_code=400, 
                    detail="Please upload a valid audio file (wav, mp3, m4a, ogg)"
                )
        
        logger.info(f"Processing audio file: {file.filename} ({file_size} bytes)")
        
        # Preprocess audio
        features, audio_length, sample_rate = preprocess_audio(content)
        
        # Make prediction
        predictions = model.predict(features)
        predicted_probs = predictions[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(predicted_probs)
        confidence = float(predicted_probs[predicted_class_idx])
        
        # Decode prediction
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        prediction_count += 1
        
        # Prepare result
        result = {
            "success": True,
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "target": target,
            "is_correct": target == predicted_class if target else None,
            "all_predictions": {
                label_encoder.inverse_transform([i])[0]: round(float(prob), 4) 
                for i, prob in enumerate(predicted_probs)
            },
            "file_info": {
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": file_size,
                "size_kb": round(file_size / 1024, 2),
                "audio_length_samples": audio_length,
                "sample_rate": sample_rate,
                "duration_seconds": round(audio_length / sample_rate, 2)
            },
            "metadata": {
                "model_version": "javanese_enhanced_retrain",
                "timestamp": datetime.now().isoformat(),
                "prediction_id": prediction_count,
                "feature_shape": features.shape
            }
        }
        
        logger.info(f"Prediction: {predicted_class} ({confidence:.4f}) for {file.filename}")
        return result
        
    except HTTPException:
        raise
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
        "deployment": "production",
        "model_loaded": model is not None,
        "encoder_loaded": label_encoder is not None
    }

@app.post("/test-upload")
async def test_upload(file: UploadFile = File(...)):
    """Test file upload functionality without prediction"""
    try:
        content = await file.read()
        
        # Basic audio info extraction
        audio_info = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(content),
            "size_kb": round(len(content) / 1024, 2)
        }
        
        # Try to get audio properties
        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(content))
            audio_info.update({
                "duration_ms": len(audio_segment),
                "duration_seconds": round(len(audio_segment) / 1000, 2),
                "channels": audio_segment.channels,
                "frame_rate": audio_segment.frame_rate,
                "sample_width": audio_segment.sample_width
            })
        except Exception as e:
            audio_info["audio_processing_note"] = f"Could not extract audio properties: {str(e)}"
        
        return {
            "success": True,
            "file_info": audio_info,
            "status": "upload_successful",
            "note": "File uploaded successfully. Use /predict endpoint for voice recognition."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload test failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting production server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
