"""
Javanese Aksara Voice Recognition API - Railway Deployment
FastAPI backend optimized for Railway platform
"""

import os
import io
import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pickle
from pydub import AudioSegment
from typing import Optional, List
import logging
from datetime import datetime
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Javanese Aksara Voice Recognition API",
    description="API for recognizing Javanese aksara characters from voice input",
    version="2.2.0",
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
model = None
label_encoder = None
start_time = datetime.now()
prediction_count = 0

# Javanese aksara classes
CLASSES = [
    'ha', 'na', 'ca', 'ra', 'ka', 'da', 'ta', 'sa', 'wa', 'la',
    'pa', 'dha', 'ja', 'ya', 'nya', 'ma', 'ga', 'ba', 'tha', 'nga'
]

def load_model_and_encoder():
    """Load TensorFlow model and label encoder"""
    global model, label_encoder
    
    try:
        # Load model
        model_path = "models/javanese_enhanced_retrain.h5"
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info(f"✅ Model loaded: {model.count_params()} parameters")
        else:
            logger.error(f"❌ Model file not found: {model_path}")
            return False
        
        # Load label encoder
        encoder_path = "models/label_encoder_retrain.pkl"
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            logger.info("✅ Label encoder loaded")
        else:
            logger.error(f"❌ Label encoder not found: {encoder_path}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model/encoder: {e}")
        return False

def extract_features(audio_data, sr=22050, target_length=2.0, n_mels=128):
    """Extract mel-spectrogram features from audio"""
    try:
        # Ensure audio is float32
        audio_data = audio_data.astype(np.float32)
        
        # Normalize length to target_length seconds
        target_samples = int(target_length * sr)
        if len(audio_data) > target_samples:
            # Take center portion
            start = (len(audio_data) - target_samples) // 2
            audio_data = audio_data[start:start + target_samples]
        else:
            # Pad with zeros
            audio_data = np.pad(audio_data, (0, target_samples - len(audio_data)))
        
        # Pre-emphasis filter
        audio_data = np.append(audio_data[0], audio_data[1:] - 0.97 * audio_data[:-1])
        
        # Apply window
        audio_data = audio_data * np.hanning(len(audio_data))
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, sr=sr, n_mels=n_mels,
            hop_length=512, win_length=2048,
            fmin=0, fmax=sr//2
        )
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # CMVN normalization
        mel_spec_normalized = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-8)
        
        return mel_spec_normalized.T  # Transpose for time-first format
        
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

def convert_audio_to_wav(audio_file):
    """Convert audio file to WAV format"""
    try:
        # Read audio file
        audio_bytes = audio_file.read()
        
        # Convert to AudioSegment
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        
        # Convert to mono, 22050 Hz
        audio = audio.set_channels(1).set_frame_rate(22050)
        
        # Export to wav bytes
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        return wav_buffer
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("🚀 Starting Javanese Aksara Recognition API...")
    success = load_model_and_encoder()
    if not success:
        logger.error("❌ Failed to load model/encoder")
    else:
        logger.info("✅ API ready for predictions")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Javanese Aksara Voice Recognition API",
        "version": "2.2.0",
        "status": "active",
        "docs": "/docs",
        "model_loaded": model is not None
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
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "Enhanced Javanese Aksara CNN",
        "version": "2.2.0",
        "accuracy": 94.67,
        "parameters": model.count_params(),
        "classes": len(CLASSES),
        "supported_aksara": CLASSES,
        "input_format": {
            "sample_rate": 22050,
            "duration": "2.0 seconds",
            "features": "128-band mel-spectrogram"
        }
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
async def predict(
    file: UploadFile = File(...),
    target: Optional[str] = Form(None)
):
    """Predict Javanese aksara from audio file"""
    global prediction_count
    
    if model is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert audio to WAV
        wav_buffer = convert_audio_to_wav(file.file)
        
        # Load audio with librosa
        audio_data, sr = librosa.load(wav_buffer, sr=22050)
        
        # Extract features
        features = extract_features(audio_data, sr)
        
        # Prepare input for model
        X = features[None, :, :, None]  # Add batch and channel dimensions
        
        # Make prediction
        predictions = model.predict(X, verbose=0)
        
        # Get prediction results
        predicted_idx = np.argmax(predictions[0])
        predicted_class = CLASSES[predicted_idx]
        confidence = float(predictions[0][predicted_idx])
        
        # Create all predictions with confidence scores
        all_predictions = []
        for i, class_name in enumerate(CLASSES):
            all_predictions.append({
                "class": class_name,
                "confidence": float(predictions[0][i])
            })
        
        # Sort by confidence
        all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Increment counter
        prediction_count += 1
        
        result = {
            "success": True,
            "prediction": predicted_class,
            "confidence": confidence,
            "target": target,
            "is_correct": predicted_class == target if target else None,
            "all_predictions": all_predictions[:5],  # Top 5
            "metadata": {
                "audio_duration": len(audio_data) / sr,
                "sample_rate": sr,
                "model_version": "2.2.0",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.3f})")
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    """Predict multiple audio files"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    for i, file in enumerate(files):
        try:
            # Process each file
            wav_buffer = convert_audio_to_wav(file.file)
            audio_data, sr = librosa.load(wav_buffer, sr=22050)
            features = extract_features(audio_data, sr)
            X = features[None, :, :, None]
            
            predictions = model.predict(X, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            predicted_class = CLASSES[predicted_idx]
            confidence = float(predictions[0][predicted_idx])
            
            results.append({
                "file_index": i,
                "filename": file.filename,
                "prediction": predicted_class,
                "confidence": confidence,
                "success": True
            })
            
        except Exception as e:
            results.append({
                "file_index": i,
                "filename": file.filename,
                "error": str(e),
                "success": False
            })
    
    return {
        "success": True,
        "total_files": len(files),
        "results": results
    }

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
        "model_status": "loaded" if model else "not_loaded",
        "supported_classes": len(CLASSES)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
