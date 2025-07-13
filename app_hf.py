import gradio as gr
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import yfinance as yf
from datetime import datetime, timedelta
import os

# Load model files
model = None
scaler_features = None
scaler_target = None

def load_model_files():
    global model, scaler_features, scaler_target
    try:
        model = load_model('stock_price_lstm_model.h5', compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        scaler_features = joblib.load('scaler_features.pkl')
        scaler_target = joblib.load('scaler_target.pkl')
        return True
    except:
        return False

# [Copy the create_features and get_stock_data functions from app.py]

def predict_stock(ticker_input):
    if not ticker_input:
        return "Please enter a stock ticker (e.g., AAPL, MSFT, GOOGL)"
    
    ticker = ticker_input.upper().strip()
    
    # [Copy the prediction logic from app.py]
    # Return formatted string with predictions
    
    return f"📊 Stock Analysis for {ticker}\\n\\nPredictions coming soon..."

# Create Gradio interface
def create_interface():
    load_model_files()
    
    iface = gr.Interface(
        fn=predict_stock,
        inputs=gr.Textbox(
            label="Stock Ticker",
            placeholder="Enter stock ticker (e.g., AAPL)",
            info="Type a stock symbol to get 5-day price predictions"
        ),
        outputs=gr.Textbox(
            label="Prediction Results",
            lines=15
        ),
        title="🚀 LSTM Stock Price Prediction",
        description="AI-powered stock price predictions using LSTM neural networks",
        examples=[["AAPL"], ["MSFT"], ["GOOGL"], ["TSLA"], ["AMZN"]],
        theme="dark"
    )
    
    return iface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()