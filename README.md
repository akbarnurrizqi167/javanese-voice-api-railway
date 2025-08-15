# 🚀 Javanese Aksara Voice Recognition API - Railway

Fast and reliable API for recognizing Javanese aksara characters from voice input, optimized for Railway deployment.

## 🎯 Features

- **Enhanced CNN Model**: 94.67% accuracy with 210K parameters
- **FastAPI Backend**: High-performance async API
- **20 Javanese Aksara**: Complete traditional character support
- **Audio Processing**: Automatic WAV conversion and normalization
- **Railway Optimized**: Lightweight dependencies and fast startup

## 🔗 Quick Links

- **API Documentation**: `/docs` (Swagger UI)
- **Health Check**: `/health`
- **Model Info**: `/model/info`
- **Supported Aksara**: `/supported-aksara`

## 📊 Supported Aksara Classes

```
ha, na, ca, ra, ka, da, ta, sa, wa, la,
pa, dha, ja, ya, nya, ma, ga, ba, tha, nga
```

## 🎤 API Usage

### Single Prediction
```bash
curl -X POST "https://your-railway-url.railway.app/predict" \
  -F "file=@audio.wav" \
  -F "target=ha"
```

### Batch Prediction
```bash
curl -X POST "https://your-railway-url.railway.app/batch-predict" \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav"
```

### Health Check
```bash
curl "https://your-railway-url.railway.app/health"
```

## 🏗️ Model Architecture

- **Type**: Enhanced CNN with GlobalAveragePooling2D
- **Input**: 128-band mel-spectrogram (22050 Hz, 2 seconds)
- **Parameters**: 210,164 trainable parameters
- **Accuracy**: 94.67% on test dataset
- **Features**: Pre-emphasis, CMVN normalization, data augmentation

## 🚀 Railway Deployment

This repository is optimized for one-click Railway deployment:

1. **Connect Repository**: Link this GitHub repo to Railway
2. **Auto-Deploy**: Railway detects Python and deploys automatically
3. **Environment**: Uses Python 3.9 with optimized dependencies
4. **Scaling**: Auto-scales based on traffic

## 📁 Project Structure

```
javanese-voice-api-railway/
├── 🚀 app/
│   └── main.py              # FastAPI application
├── 🤖 models/
│   ├── javanese_enhanced_retrain.h5
│   └── label_encoder_retrain.pkl
├── 📋 requirements.txt      # Lightweight dependencies
├── ⚙️ railway.json          # Railway configuration
├── 🐳 Dockerfile           # Container setup
└── 📚 README.md            # This file
```

## 🎯 Performance

- **Cold Start**: ~10-15 seconds (model loading)
- **Prediction**: ~200-500ms per audio file
- **Throughput**: 10+ requests/second
- **Memory**: ~2GB RAM usage
- **Storage**: ~5MB model files

## 🔧 Environment Variables

Railway automatically handles:
- `PORT`: Application port (default: 8000)
- `NIXPACKS_*`: Build configuration

## 📈 API Response Example

```json
{
  "success": true,
  "prediction": "ha",
  "confidence": 0.954,
  "target": "ha",
  "is_correct": true,
  "all_predictions": [
    {"class": "ha", "confidence": 0.954},
    {"class": "na", "confidence": 0.023},
    {"class": "ca", "confidence": 0.015}
  ],
  "metadata": {
    "audio_duration": 2.0,
    "sample_rate": 22050,
    "model_version": "2.2.0",
    "timestamp": "2025-08-15T23:10:00"
  }
}
```

## 🛠️ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn app.main:app --reload --port 8000

# Test API
curl http://localhost:8000/health
```

## 📊 Model Information

- **Training Dataset**: Enhanced with data augmentation
- **Validation**: Stratified split with early stopping
- **Regularization**: Dropout, BatchNorm, L2 regularization
- **Optimizer**: Adam with learning rate scheduling
- **Loss**: Categorical crossentropy with class weights

---

**Developed for Railway Platform** ⚡ **Ready for Production** 🚀
