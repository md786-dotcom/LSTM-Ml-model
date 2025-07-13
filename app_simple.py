from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-this')

def get_stock_data(ticker, period='1mo'):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker.upper())
        df = stock.history(period=period)
        if df.empty:
            return None
        df.reset_index(inplace=True)
        return df
    except:
        return None

def simple_prediction(ticker):
    """Simple prediction without ML model"""
    df = get_stock_data(ticker)
    if df is None or len(df) < 5:
        return None
    
    # Simple moving average based prediction
    current_price = df['Close'].iloc[-1]
    ma5 = df['Close'].tail(5).mean()
    ma20 = df['Close'].tail(20).mean() if len(df) >= 20 else ma5
    
    # Simple trend prediction
    trend = (ma5 - ma20) / ma20 * 100 if ma20 > 0 else 0
    
    # Generate predictions
    predictions = []
    for i in range(5):
        # Simple linear projection
        change = trend * (i + 1) * 0.2  # Dampen the change
        predicted_price = current_price * (1 + change / 100)
        predictions.append(predicted_price)
    
    return {
        'ticker': ticker.upper(),
        'current_price': round(current_price, 2),
        'predictions': predictions,
        'trend': trend,
        'recommendation': 'BUY' if trend > 1 else 'SELL' if trend < -1 else 'HOLD'
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_message = data.get('message', '').strip().upper()
    
    # Extract ticker
    ticker = None
    words = user_message.split()
    for word in words:
        if len(word) <= 5 and word.isalpha():
            ticker = word
            break
    
    if not ticker:
        return jsonify({
            'response': "Please specify a stock ticker. For example: 'Predict AAPL'",
            'error': False
        })
    
    # Get prediction
    result = simple_prediction(ticker)
    
    if not result:
        return jsonify({
            'response': f"Sorry, couldn't fetch data for {ticker}. Please try again.",
            'error': True
        })
    
    # Format response
    response = f"""📊 **Stock Analysis for {result['ticker']}**

💵 Current Price: ${result['current_price']}

📈 5-Day Prediction (Simple Moving Average):
"""
    
    for i, price in enumerate(result['predictions']):
        date = (datetime.now() + timedelta(days=i+1)).strftime('%b %d')
        response += f"\n• {date}: ${price:.2f}"
    
    trend_emoji = "📈" if result['trend'] > 0 else "📉"
    response += f"\n\n{trend_emoji} Trend: {result['trend']:+.1f}%"
    response += f"\n\n💡 Recommendation: **{result['recommendation']}**"
    response += "\n\n*Note: This is a simple moving average prediction, not ML-based.*"
    
    return jsonify({
        'response': response,
        'error': False
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'type': 'simple'})

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5001))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    app.run(host=host, port=port, debug=debug)