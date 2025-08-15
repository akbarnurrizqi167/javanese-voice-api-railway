# Railway Deployment Guide

## 🚀 Quick Deploy Options

### Option 1: Light Version (Recommended for First Test)
```bash
# Use light configuration
cp railway-light.json railway.json
cp requirements-ultra-light.txt requirements.txt
```

This version includes:
- ✅ Basic FastAPI with file upload
- ✅ Mock predictions for testing
- ✅ Minimal dependencies
- ✅ Fast deployment (~2-3 minutes)

### Option 2: Full ML Version
```bash
# Use full configuration  
cp railway.json railway.json
cp requirements.txt requirements.txt
```

This version includes:
- ✅ Complete ML model (TensorFlow)
- ✅ Real predictions with 94.67% accuracy
- ✅ Full feature extraction
- ⚠️ Longer deployment (~10-15 minutes)
- ⚠️ Higher memory usage

## 📋 Railway Setup Steps

1. **Connect Repository**
   ```
   https://github.com/akbarnurrizqi167/javanese-voice-api-railway
   ```

2. **Configure Build**
   - Build Method: Dockerfile
   - Dockerfile Path: `Dockerfile.light` (for light) or `Dockerfile` (for full)

3. **Environment Variables**
   ```
   PYTHONUNBUFFERED=1
   TF_CPP_MIN_LOG_LEVEL=2  # (full version only)
   ```

4. **Health Check**
   - Endpoint: `/health`
   - Timeout: 30s

## 🔧 API Endpoints

### Light Version
- `GET /` - API info
- `GET /health` - Health check
- `GET /supported-aksara` - List of aksara
- `POST /test-upload` - File upload test
- `POST /mock-predict` - Mock prediction

### Full Version  
- All light endpoints plus:
- `POST /predict` - Real ML prediction
- `POST /batch-predict` - Batch predictions
- `GET /model/info` - Model details

## 🧪 Testing

### Local Test (Light Version)
```bash
PORT=8000 python app/main_light.py
curl http://localhost:8000/health
```

### Local Test (Full Version)
```bash
PORT=8000 python app/main.py
curl http://localhost:8000/health
```

## 📊 Expected Response Times

| Version | Cold Start | Warm Request | Build Time |
|---------|------------|--------------|------------|
| Light   | ~2-3s      | ~100-200ms   | ~2-3 min   |
| Full    | ~10-15s    | ~500-1000ms  | ~10-15 min |

## 🔍 Troubleshooting

### Build Failures
1. **Dependency Errors**: Use light version first
2. **Memory Issues**: Enable swap or use lighter dependencies
3. **Timeout**: Increase Railway timeout settings

### Runtime Issues  
1. **Health Check Fails**: Check `/health` endpoint manually
2. **Cold Starts**: First request may be slow
3. **Memory Usage**: Monitor Railway metrics

## 📈 Scaling Options

1. **Start Light**: Deploy light version first
2. **Test Endpoints**: Verify all functionality
3. **Upgrade**: Switch to full ML version
4. **Optimize**: Add caching and optimizations

## 🎯 Success Criteria

✅ **Light Version**: 
- Health check returns 200
- File upload works
- Mock predictions return

✅ **Full Version**:
- All light features plus
- Real ML predictions with confidence scores
- Model info endpoint returns details

Railway URL: `https://your-app-name.railway.app`
