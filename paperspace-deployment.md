# Paperspace Gradient Deployment Guide

## Steps to Deploy Your Stock Prediction Chatbot on Paperspace Gradient

### 1. Prepare Your Files
Ensure you have these files from your Kaggle notebook:
- `stock_price_lstm_model.h5` - Your trained model
- `scaler_features.pkl` - Feature scaler
- `scaler_target.pkl` - Target scaler

### 2. Create a Paperspace Gradient Account
1. Go to [gradient.paperspace.com](https://gradient.paperspace.com)
2. Sign up for Gradient Pro

### 3. Create a New Project
1. Click "Create Project"
2. Name it "Stock Prediction Chatbot"

### 4. Deploy Using Deployments

#### Option A: Using Git Repository (Recommended)
1. Push your code to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. In Paperspace Gradient:
   - Go to Deployments → Create Deployment
   - Choose "From GitHub"
   - Connect your repository
   - Set:
     - Container: Custom (use our Dockerfile)
     - Port: 5001
     - Machine: C4 (CPU) or P4000 (GPU)
     - Replicas: 1-3 for high availability

#### Option B: Using Paperspace CLI
1. Install Gradient CLI:
   ```bash
   pip install gradient
   gradient apikey set YOUR_API_KEY
   ```

2. Deploy directly:
   ```bash
   gradient deployments create \
     --name "stock-prediction-bot" \
     --projectId YOUR_PROJECT_ID \
     --image Dockerfile \
     --port 5001 \
     --machineType C4 \
     --replicas 1
   ```

### 5. Environment Variables (Optional)
Add these in Deployment settings if needed:
- `FLASK_ENV`: production
- `TF_CPP_MIN_LOG_LEVEL`: 2

### 6. Custom Domain (Optional)
1. Go to Deployment settings
2. Add custom domain
3. Update DNS records as instructed

### 7. Monitor Your Deployment
- Check logs: Deployments → Your deployment → Logs
- Monitor metrics: CPU, Memory, Request count
- Set up alerts for downtime

## Local Testing Before Deployment

1. Build Docker image:
   ```bash
   docker build -t stock-bot .
   ```

2. Run locally:
   ```bash
   docker run -p 5001:5001 stock-bot
   ```

3. Test at http://localhost:5001

## Troubleshooting

### Model files too large for Git?
Use Git LFS:
```bash
git lfs track "*.h5"
git lfs track "*.pkl"
git add .gitattributes
git add stock_price_lstm_model.h5 scaler_features.pkl scaler_target.pkl
git commit -m "Add model files with LFS"
git push
```

### Out of memory errors?
- Upgrade to a larger machine type (C5 or P4000)
- Reduce model batch size in predictions
- Implement model caching

### Slow predictions?
- Enable GPU support (change to P4000 machine)
- Implement Redis caching for frequent tickers
- Use model quantization to reduce size

## Cost Optimization
- Use CPU machines (C4/C5) instead of GPU for inference
- Set up auto-scaling based on traffic
- Use spot instances for development

## Security Best Practices
1. Don't expose your API directly - use Paperspace's built-in authentication
2. Rate limit requests to prevent abuse
3. Validate all user inputs
4. Use HTTPS only (Paperspace provides this)

Your app will be available at:
`https://YOUR_DEPLOYMENT_ID.gradient.run`