# LSTM Stock Price Prediction Model

A deep learning-based stock price prediction system using LSTM neural networks with a beautiful web interface.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.0-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)

## 🚀 Features

- **LSTM Neural Network**: Advanced time series prediction using 3-layer LSTM architecture
- **Beautiful Chat Interface**: Modern, responsive web UI with real-time predictions
- **Stock Analysis**: Predicts stock prices for the next 5 days
- **Buy/Sell/Hold Recommendations**: AI-powered trading signals
- **Real-time Data**: Fetches latest stock data using yfinance
- **Docker Support**: Easy deployment with containerization

## 📸 Screenshots

### Chat Interface
The application features a modern dark-themed chat interface where users can interact with the AI to get stock predictions.

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/LSTM-Ml-model.git
cd LSTM-Ml-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download model files:
   - Download the trained model files from the Kaggle notebook output:
     - `stock_price_lstm_model.h5`
     - `scaler_features.pkl`
     - `scaler_target.pkl`
   - Place these files in the project root directory

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5001`

## 🐳 Docker Deployment

Build and run with Docker:

```bash
# Build the image
docker build -t stock-prediction-bot .

# Run the container
docker run -p 5001:5001 stock-prediction-bot
```

## 📊 Model Details

### Architecture
- **Input Layer**: 60 timesteps, 13 features
- **LSTM Layers**: 3 layers (128, 64, 32 units) with dropout
- **Dense Layers**: 2 layers (64, 32 units)
- **Output Layer**: 5 predictions (next 5 days)

### Features Used
- OHLCV data (Open, High, Low, Close, Volume)
- Technical indicators:
  - Moving Averages (5, 20, 50 days)
  - RSI (Relative Strength Index)
  - Volatility
  - Price ratios

### Training Data
The model was trained on historical stock market data from NYSE, NASDAQ, and NYSE MKT exchanges.

## 🚀 Deployment on Paperspace Gradient

See [paperspace-deployment.md](paperspace-deployment.md) for detailed deployment instructions on Paperspace Gradient Pro.

## 💬 Usage

1. Type a stock ticker in the chat (e.g., "Predict AAPL")
2. Or click on popular stock buttons
3. Get AI-powered predictions and recommendations

### Example Commands
- "Predict AAPL"
- "What will TSLA do?"
- "Analyze MSFT stock"

## ⚠️ Disclaimer

This tool is for educational purposes only. Stock predictions are based on historical data and should not be used as the sole basis for investment decisions. Always do your own research and consult with financial advisors.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: [Price Volume Data for All US Stocks & ETFs](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)
- Built with TensorFlow, Flask, and yfinance
- Trained using Kaggle Notebooks