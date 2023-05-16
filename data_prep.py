from pycoingecko import CoinGeckoAPI as cg
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np


def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    up_days = delta.copy()
    up_days[delta<=0]=0.0
    down_days = abs(delta.copy())
    down_days[delta>0]=0.0
    RS_up = up_days.rolling(window).mean()
    RS_down = down_days.rolling(window).mean()
    rsi= 100-100/(1+RS_up/RS_down)
    return rsi


def calculate_volatility(data, window=14):
    volatility = data['Close'].diff().rolling(window).std()
    return volatility


def chooseData(coin):
    start_date = "2015-01-01"
    cg_api = cg()
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    today = datetime.now()
    days = (today - datetime.strptime(start_date, "%Y-%m-%d")).days
    coin_id = coin.lower()
    data = cg_api.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days, interval='daily')
    new_data = pd.DataFrame(data['prices'], columns=['timestamp', 'Close'])
    new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], unit='ms')
    new_data.set_index('timestamp', inplace=True)
    new_data = new_data[new_data.index >= start_datetime]
    
    # Calculate RSI and Volatility
    new_data['RSI'] = calculate_rsi(new_data)
    new_data['Volatility'] = calculate_volatility(new_data)
    
    # Drop rows with NaN values
    new_data = new_data.dropna()

    # Normalize the features separately
    scaler_close = MinMaxScaler()
    scaler_rsi = MinMaxScaler()
    scaler_volatility = MinMaxScaler()

    new_data['Close'] = scaler_close.fit_transform(new_data['Close'].values.reshape(-1,1))
    new_data['RSI'] = scaler_rsi.fit_transform(new_data['RSI'].values.reshape(-1,1))
    new_data['Volatility'] = scaler_volatility.fit_transform(new_data['Volatility'].values.reshape(-1,1))
    
    return scaler_close, new_data.values


def create_sequences(data, seq_length):
    inputs = []
    labels = []

    # New variable to store the last sequence

    for i in range(len(data) - seq_length):
        inputs.append(data[i:i + seq_length])
        labels.append(data[i + seq_length, 0])

        if i == len(data) - seq_length - 1:
            last_sequence = data[i + 1:i + 1 + seq_length, 0]

    inputs = np.array(inputs).reshape(-1, seq_length, data.shape[1])
    labels = np.array(labels)

    return inputs, labels


def sortData(data, train_ratio=0.95):
    seq_length = 128
    inputs, labels = create_sequences(data, seq_length)
    total_size = len(inputs)
    train_size = int(total_size * train_ratio)

    train_inputs = torch.tensor(inputs[:train_size], dtype=torch.float32)
    train_labels = torch.tensor(labels[:train_size], dtype=torch.float32)

    test_inputs = torch.tensor(inputs[train_size:], dtype=torch.float32)
    test_labels = torch.tensor(labels[train_size:], dtype=torch.float32)
    
    return train_inputs, train_labels, test_inputs, test_labels


def prep_tomorrow_price(tomorrow_price):
    if tomorrow_price < 10:
        tomorrow_price = round(tomorrow_price, 4)
    else:
        tomorrow_price = round(tomorrow_price, 2)
    return tomorrow_price
