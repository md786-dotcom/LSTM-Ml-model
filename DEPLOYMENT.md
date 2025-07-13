# 🚀 Deployment Guide

This guide covers deploying the LSTM Stock Prediction app on various free platforms.

## Render (Recommended for Flask apps)

### Prerequisites
- GitHub account with the repository
- Model files (`*.h5`, `*.pkl`) committed via Git LFS

### Steps

1. **Prepare your repository**
   ```bash
   # Make sure all files are committed
   git add .
   git commit -m "Add deployment configurations"
   git push origin main
   ```

2. **Sign up for Render**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

3. **Create a new Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select `LSTM-Ml-model` repository

4. **Configure the service**
   - Name: `stock-prediction-bot`
   - Environment: `Python`
   - Build Command: (auto-detected from render.yaml)
   - Start Command: (auto-detected from render.yaml)

5. **Environment Variables** (auto-configured from render.yaml)
   - `SECRET_KEY`: Auto-generated
   - `FLASK_ENV`: production
   - Model paths are pre-configured

6. **Deploy**
   - Click "Create Web Service"
   - Wait for build and deployment (~5-10 minutes first time)

Your app will be available at: `https://stock-prediction-bot.onrender.com`

### Notes
- Free tier spins down after 15 minutes of inactivity
- First request after spin-down takes ~30 seconds
- Perfect for portfolio/demo projects

## Alternative Platforms

### Streamlit Cloud (Easiest for Python)
1. Use `app_streamlit.py` instead
2. Push to GitHub
3. Go to [share.streamlit.io](https://share.streamlit.io)
4. Deploy directly from GitHub

### Railway
1. Install Railway CLI: `npm i -g @railway/cli`
2. Run: `railway login` then `railway up`
3. Configure environment variables in dashboard

### Hugging Face Spaces
1. Create account at [huggingface.co](https://huggingface.co)
2. Create new Space (Gradio)
3. Upload `app_hf.py` and model files
4. Auto-deploys on push

## Troubleshooting

### Model files too large?
- Use Git LFS (see setup_git_lfs.sh)
- Or host on Google Drive and download in app startup

### Out of memory?
- Reduce model complexity
- Use smaller batch sizes
- Consider Hugging Face Spaces (more generous resources)

### Slow predictions?
- Normal on free tiers
- First prediction after cold start is slower
- Consider caching predictions

## Environment Variables

Copy `.env.example` to `.env` for local development:
```bash
cp .env.example .env
# Edit .env with your values
```

Never commit `.env` to GitHub!