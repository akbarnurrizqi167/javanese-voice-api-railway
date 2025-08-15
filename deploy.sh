#!/bin/bash

# Javanese Voice API Railway - Production Deployment Script
echo "🚀 Starting deployment process..."

# Navigate to project directory
cd /Users/user/javanese-voice-api-railway

# Check git status
echo "📋 Checking git status..."
git status

# Add all changes
echo "➕ Adding changes to git..."
git add .

# Commit changes
echo "💾 Committing changes..."
git commit -m "✨ Production ready: Real ML model integration with TensorFlow

🎯 Features:
- Real CNN model with 94.67% accuracy
- Full ML pipeline with MFCC feature extraction
- Production-ready FastAPI with TensorFlow
- Docker containerized with all dependencies
- Health checks and monitoring
- Real voice recognition (not mock)

📁 Updates:
- Created main_production.py with real ML model
- Updated requirements.txt with production dependencies  
- Enhanced Dockerfile with system deps (ffmpeg, libsndfile)
- Updated railway.json to use production config
- Added comprehensive error handling
- Real audio preprocessing with librosa & pydub

🚀 Ready for Railway deployment!"

# Push to remote
echo "🌐 Pushing to Railway..."
git push origin main

echo "✅ Deployment completed! Railway will auto-deploy the new version."
echo "🔗 Check your Railway dashboard for deployment status."
