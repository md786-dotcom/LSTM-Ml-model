#!/bin/bash

echo "=== Git LFS Setup Script ==="
echo ""
echo "This script will help you set up Git LFS for the model files."
echo ""

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Git LFS is not installed!"
    echo ""
    echo "Please install Git LFS first:"
    echo "  - macOS: brew install git-lfs"
    echo "  - Or download from: https://git-lfs.github.com/"
    echo ""
    exit 1
fi

echo "✅ Git LFS is installed"
echo ""

# Initialize Git LFS
echo "Initializing Git LFS..."
git lfs install

# Create .gitattributes for LFS
echo "Setting up Git LFS tracking..."
git lfs track "*.h5"
git lfs track "*.pkl"

# Add .gitattributes to git
git add .gitattributes
git commit -m "Add Git LFS tracking for model files"

echo ""
echo "✅ Git LFS is now configured!"
echo ""
echo "Next steps:"
echo "1. Add your model files to the repository:"
echo "   git add stock_price_lstm_model.h5 scaler_features.pkl scaler_target.pkl"
echo "   git commit -m 'Add trained model files'"
echo ""
echo "2. Push to GitHub:"
echo "   git push origin main"
echo ""
echo "Note: Large files will be uploaded to Git LFS storage"