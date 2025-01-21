import os
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler

MODEL_PATH = 'enhanced_lstm_model.h5'
model = load_model(MODEL_PATH)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def fetch_latest_forex(api_url, params):
    response = requests.get(api_url, params=params)
    data = response.json()
    if 'values' not in data:
        print("Error fetching Forex data.", data)
        return None
    return data

def fetch_latest_news(api_key, latest_time):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&FOREX:USD&time_from={latest_time}&limit=1000&apikey={api_key}"
    print("Alpha Vantage Request URL:", url)  
    response = requests.get(url)
    data = response.json()
    if 'feed' not in data or not data['feed']:
        print("No news data fetched. API Response:", data)
        return None
    return data

def get_time_two_days_ago_alphavantage_format():
    two_days_ago = datetime.now(timezone.utc) - timedelta(days=2)  
    return two_days_ago.strftime("%Y%m%dT%H%M")  

def calculate_rsi(data, window=14):
    data = pd.to_numeric(data, errors='coerce')  
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = data.ewm(span=fast_period, min_periods=1, adjust=False).mean()
    slow_ema = data.ewm(span=slow_period, min_periods=1, adjust=False).mean()
    
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, min_periods=1, adjust=False).mean()
    return macd, signal

def merge_data(forex_data, news_data):
    forex_df = pd.DataFrame.from_records(forex_data['values'])
    forex_df['Datetime'] = pd.to_datetime(forex_df['datetime'])
    forex_df.drop(columns=['datetime'], inplace=True)
    
    forex_df['close'] = pd.to_numeric(forex_df['close'], errors='coerce')
    forex_df['open'] = pd.to_numeric(forex_df['open'], errors='coerce')
    forex_df['high'] = pd.to_numeric(forex_df['high'], errors='coerce')
    forex_df['low'] = pd.to_numeric(forex_df['low'], errors='coerce')

    news_records = []
    for item in news_data['feed']:
        news_records.append({
            "Datetime": pd.to_datetime(item['time_published']),
            "Overall Sentiment Score": item.get('overall_sentiment_score', 0),
            "Overall Sentiment Label": item.get('overall_sentiment_label', '')
        })
    news_df = pd.DataFrame(news_records)

    merged_df = pd.merge(forex_df, news_df, on='Datetime', how='inner')

    merged_df['5-Minute MA'] = merged_df['close'].rolling(window=5).mean()
    merged_df['10-Minute MA'] = merged_df['close'].rolling(window=10).mean()

    merged_df['RSI'] = calculate_rsi(merged_df['close'])
    merged_df['MACD'], merged_df['MACD_Signal'] = calculate_macd(merged_df['close'])

    merged_df['Momentum'] = merged_df['close'].diff()  
    merged_df['Volatility'] = merged_df['high'] - merged_df['low']

    merged_df = merged_df.fillna(0) 

    return merged_df

def update_predictions(merged_df, scaler, time_step=10):
    features = merged_df[['open', 'high', 'low', 'Overall Sentiment Score', '5-Minute MA', '10-Minute MA', 'Momentum', 'Volatility', 'RSI', 'MACD']]

    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].fillna(features[numeric_cols].mean())

    scaled_features = scaler.transform(features)

    latest_sequence = scaled_features[-time_step:].reshape(1, time_step, scaled_features.shape[1])

    
    prediction = model.predict(latest_sequence)
    predicted_label = np.argmax(prediction, axis=1)[0] 

    return predicted_label

if __name__ == "__main__":
    FOREX_FILE = 'forex_data.csv'
    NEWS_FILE = 'news_sentiment.csv'

    
    forex_df = pd.read_csv(FOREX_FILE)
    news_df = pd.read_csv(NEWS_FILE)

    latest_forex_time = forex_df['Datetime'].max()
    latest_news_time = get_time_two_days_ago_alphavantage_format()

    api_url_forex = "https://api.twelvedata.com/time_series"
    params_forex = {
        "symbol": "USD/INR",
        "interval": "1min",
        "apikey": "70f234ce0c03409a974bef5ec5f11bce",
        "start_date": latest_forex_time
    }
    forex_updates = fetch_latest_forex(api_url_forex, params_forex)

    api_key_news = "54257QBPQUWW5HEZ"
    news_updates = fetch_latest_news(api_key_news, latest_news_time)

    if forex_updates and news_updates:
        merged_df = merge_data(forex_updates, news_updates)
        merged_df.to_csv('updated_merged_data.csv', index=False)
        print("Data merged and saved to updated_merged_data.csv")

        merged_df = pd.read_csv('updated_merged_data.csv')

        scaler = MinMaxScaler()
        scaler.fit(merged_df[['open', 'high', 'low', 'Overall Sentiment Score', '5-Minute MA', '10-Minute MA', 'Momentum', 'Volatility', 'RSI', 'MACD']])

        predicted_label = update_predictions(merged_df, scaler)
        print("Predicted Label:", predicted_label)

        label_mapping = {
            0: "Transfer Now", 
            1: "Wait", 
            2: "Uncertain", 
            3: "Urgent Transfer Now"
        }
        predicted_label_mapped = label_mapping.get(predicted_label, "Unknown")

        print("Mapped Prediction:", predicted_label_mapped)
        exit()  
    else:
        print("No new data to update.")
