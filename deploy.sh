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
git commit -m "🔧 Fix TensorFlow model loading compatibility issues

🐛 Fixed Issues:
- TensorFlow model loading with batch_shape compatibility
- Added fallback model architecture creation
- Updated feature extraction to mel-spectrogram (87,128,1)
- Added TensorFlow environment optimization
- Enhanced error handling for model prediction
- Added verbose=0 to reduce prediction logs

🎯 Features:
- Real CNN model with fallback handling
- Mel-spectrogram features matching model input shape
- Production-ready with comprehensive error handling
- TensorFlow CPU optimization for Railway
- Multiple model loading strategies

📊 Technical Updates:
- Model input: (87, 128, 1) mel-spectrogram
- TensorFlow warnings suppression
- Graceful fallback when model fails
- Enhanced audio preprocessing pipeline

🚀 Ready for Railway deployment with improved stability!"

# Push to remote
echo "🌐 Pushing to Railway..."
git push origin main

echo "✅ Deployment completed! Railway will auto-deploy the new version."
echo "🔗 Check your Railway dashboard for deployment status."
echo "🎯 Fixed TensorFlow compatibility issues - model should load properly now!"
