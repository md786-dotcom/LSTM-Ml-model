from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Optimize TensorFlow for low memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Import TensorFlow with optimizations
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Force CPU usage
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-this')

# Global variables for model
model = None
scaler_features = None
scaler_target = None
model_loaded = False

def load_model_files():
    """Load model with memory optimization"""
    global model, scaler_features, scaler_target, model_loaded
    
    if model_loaded:
        return True
    
    # Get file paths from environment variables or use defaults
    model_files = {
        'model': os.getenv('MODEL_PATH', 'stock_price_lstm_model.h5'),
        'scaler_features': os.getenv('SCALER_FEATURES_PATH', 'scaler_features.pkl'),
        'scaler_target': os.getenv('SCALER_TARGET_PATH', 'scaler_target.pkl')
    }
    
    missing_files = []
    for name, filename in model_files.items():
        if not os.path.exists(filename):
            missing_files.append(filename)
    
    if missing_files:
        print(f"ERROR: Missing model files: {missing_files}")
        return False
    
    try:
        print("Loading model files with optimization...")
        
        # Load scalers first (smaller memory footprint)
        scaler_features = joblib.load(model_files['scaler_features'])
        scaler_target = joblib.load(model_files['scaler_target'])
        
        # Load model with custom objects to avoid memory spike
        from tensorflow.keras.models import load_model as tf_load_model
        model = tf_load_model(model_files['model'], compile=False)
        
        # Compile with minimal options
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
        
        model_loaded = True
        print("Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        model_loaded = False
        return False

# Feature engineering functions
def create_features(df):
    """Create features for prediction"""
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def get_stock_data(ticker, period='6mo'):
    """Fetch stock data using yfinance"""
    try:
        print(f"Fetching data for {ticker}...")
        stock = yf.Ticker(ticker.upper())
        df = stock.history(period=period)
        
        if len(df) < 100:
            df = stock.history(period='1y')
        
        if len(df) < 100:
            df = stock.history(period='max')
            if len(df) > 200:
                df = df.tail(200)
            
        if df.empty:
            return None
            
        print(f"Successfully fetched {len(df)} records for {ticker}")
        df.reset_index(inplace=True)
        return df
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def predict_stock_price(ticker, days_ahead=5):
    """Predict stock prices for the given ticker"""
    print(f"\n=== Starting prediction for {ticker} ===")
    
    # Lazy load model on first prediction
    if not model_loaded:
        if not load_model_files():
            return None, "Model not loaded. Please ensure model files are in the current directory."
    
    if model is None or scaler_features is None or scaler_target is None:
        return None, "Model not available. This might be due to memory constraints."
    
    # Get stock data
    df = get_stock_data(ticker)
    if df is None:
        return None, "Could not fetch stock data. Please check if the ticker symbol is valid or try again later."
    
    print(f"Raw data shape: {df.shape}")
    
    if len(df) < 100:
        return None, f"Insufficient data for prediction. Only {len(df)} days available, need at least 100."
    
    # Create features
    df = create_features(df)
    df = df.dropna()
    
    print(f"Data shape after features: {df.shape}")
    
    if len(df) < 60:
        return None, "Insufficient data after feature engineering"
    
    # Prepare data for prediction
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                      'MA_5', 'MA_20', 'High_Low_Ratio', 'Close_Open_Ratio',
                      'Daily_Return', 'Volatility', 'Volume_Ratio', 'RSI']
    
    # Get last 60 days of data
    recent_data = df[feature_columns].tail(60).values
    
    # Get current price
    current_price = df['Close'].iloc[-1]
    
    try:
        # Scale the data
        recent_scaled = scaler_features.transform(recent_data)
        
        # Reshape for LSTM
        X = recent_scaled.reshape(1, 60, len(feature_columns))
        
        # Make prediction
        prediction_scaled = model.predict(X, verbose=0)
        
        # Convert to percentage changes
        base_prediction = 0.5
        percentage_changes = (prediction_scaled[0] - base_prediction) * 10
        
        # Apply percentage changes to current price
        predicted_prices = current_price * (1 + percentage_changes / 100)
        
        # Create prediction data
        dates = pd.date_range(start=df['Date'].iloc[-1] + timedelta(days=1), 
                             periods=days_ahead, freq='D')
        
        predictions_df = pd.DataFrame({
            'Date': dates,
            'Predicted_Price': predicted_prices[:days_ahead],
            'Current_Price': current_price
        })
        
        # Calculate metrics
        first_prediction = predicted_prices[0]
        price_change = first_prediction - current_price
        price_change_pct = (price_change / current_price) * 100
        
        return {
            'ticker': ticker.upper(),
            'current_price': round(current_price, 2),
            'predictions': predictions_df.to_dict('records'),
            'price_change': round(price_change, 2),
            'price_change_pct': round(price_change_pct, 2),
            'recommendation': 'BUY' if price_change > 0 else 'SELL' if price_change < -1 else 'HOLD'
        }, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_message = data.get('message', '').strip()
    
    # Parse user message for stock ticker
    words = user_message.upper().split()
    ticker = None
    
    for i, word in enumerate(words):
        if word in ['PREDICT', 'FORECAST', 'ANALYZE']:
            if i + 1 < len(words):
                ticker = words[i + 1]
                break
        elif len(word) <= 5 and word.isalpha():
            ticker = word
    
    if not ticker:
        return jsonify({
            'response': "I can help you predict stock prices! Please specify a stock ticker. For example: 'Predict AAPL' or 'What will TSLA do?'",
            'error': False
        })
    
    # Make prediction
    result, error = predict_stock_price(ticker)
    
    if error:
        return jsonify({
            'response': f"Sorry, I couldn't analyze {ticker}. {error}",
            'error': True
        })
    
    # Format response
    response = f"""📊 **Stock Analysis for {result['ticker']}**

💵 Current Price: ${result['current_price']}

📈 5-Day Prediction:
"""
    
    for i, pred in enumerate(result['predictions']):
        date = pd.to_datetime(pred['Date']).strftime('%b %d')
        price = round(pred['Predicted_Price'], 2)
        response += f"\n• {date}: ${price}"
    
    response += f"\n\n📊 Expected Change: ${result['price_change']} ({result['price_change_pct']:+.1f}%)"
    response += f"\n\n💡 Recommendation: **{result['recommendation']}**"
    
    if result['recommendation'] == 'BUY':
        response += "\n\n✅ The model predicts an upward trend. Consider buying."
    elif result['recommendation'] == 'SELL':
        response += "\n\n⚠️ The model predicts a downward trend. Consider selling."
    else:
        response += "\n\n⏸️ The model predicts minimal change. Hold your position."
    
    response += "\n\n📝 *Note: This model was trained on historical data and predictions are based on relative price movements. Always do your own research before making investment decisions.*"
    
    return jsonify({
        'response': response,
        'data': result,
        'error': False
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model_loaded,
        'memory_optimized': True
    })

# Don't load model on startup in production - wait for first request
if __name__ == '__main__':
    # Only load model in development
    if os.getenv('FLASK_ENV') == 'development':
        load_model_files()
    
    # Get configuration from environment
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5001))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    app.run(host=host, port=port, debug=debug)