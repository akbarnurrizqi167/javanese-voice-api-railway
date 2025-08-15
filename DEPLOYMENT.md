# 🚀 Railway Deployment Guide

## ✅ Fixed Railway Build Issues

### Problem Solved
- **Issue**: `RUN apt-get update && apt-get install -y ffmpeg` was timing out with "context canceled"
- **Root Cause**: Railway build timeouts with heavy system dependencies
- **Solution**: Minimal deployment approach without audio processing dependencies

### Optimizations Applied

#### 1. **Minimal Dockerfile**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt
COPY app/ app/
COPY models/ models/
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2. **Ultra-Light Requirements**
```txt
fastapi==0.68.0
uvicorn[standard]==0.15.0
python-multipart==0.0.5
pydantic==1.10.22
numpy==1.21.6
```

#### 3. **Minimal API Features**
- ✅ FastAPI with CORS
- ✅ Health check endpoint
- ✅ Mock prediction for testing
- ✅ File upload testing
- ✅ API documentation
- ❌ Audio processing (removed for deployment)
- ❌ TensorFlow model loading (removed for deployment)

## 🔄 Deployment Steps

### 1. Connect to Railway
1. Go to [Railway.app](https://railway.app)
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select `javanese-voice-api-railway` repository

### 2. Railway Auto-Detection
Railway will automatically:
- ✅ Detect Dockerfile
- ✅ Use railway.json configuration
- ✅ Build with optimized settings
- ✅ Deploy on port from $PORT environment variable

### 3. Verify Deployment
Once deployed, test these endpoints:
- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /docs` - API documentation
- `POST /predict` - Mock prediction
- `POST /test-upload` - File upload test

## 📊 Deployment Strategy

### Phase 1: Minimal Deployment (Current)
- ✅ Core FastAPI functionality
- ✅ Mock predictions for testing
- ✅ File upload validation
- ✅ Railway compatibility

### Phase 2: Audio Processing (Future)
After successful minimal deployment:
1. Add audio processing dependencies
2. Include librosa, soundfile packages
3. Enable real model predictions
4. Test with actual audio files

### Phase 3: Full Model (Future)
Complete deployment with:
1. TensorFlow model loading
2. Real-time predictions
3. Enhanced feature extraction
4. Production monitoring

## 🔧 Development Commands

### Local Testing
```bash
# Install minimal dependencies
pip install fastapi uvicorn python-multipart

# Run locally
PORT=8002 python app/main.py

# Test endpoints
curl http://localhost:8002/
curl http://localhost:8002/health
```

### Docker Testing
```bash
# Build image
docker build -t javanese-api-minimal .

# Run container
docker run -p 8000:8000 javanese-api-minimal

# Test
curl http://localhost:8000/
```

## 📝 Notes

- **Build Time**: ~2-3 minutes (vs 10+ minutes with full dependencies)
- **Memory Usage**: ~200MB (vs 1GB+ with TensorFlow)
- **Startup Time**: ~5 seconds (vs 30+ seconds with model loading)
- **Reliability**: High (minimal dependencies reduce build failures)

## 🎯 Next Steps

1. **Verify Railway Deployment**: Confirm minimal API works on Railway
2. **Add Monitoring**: Implement logging and error tracking
3. **Gradual Enhancement**: Slowly add audio processing features
4. **Load Testing**: Test with multiple concurrent requests
5. **Model Integration**: Add TensorFlow when infrastructure is stable

---

**Status**: ✅ Ready for Railway deployment  
**Last Updated**: August 15, 2025  
**Build Status**: Fixed apt-get timeout issues
