from pycoingecko import CoinGeckoAPI as cg
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np


def update_price_data(coin, price_data, start_date):
    cg_api = cg()  # Add this line to create an instance of CoinGeckoAPI
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")  # Convert start_date to datetime object
    today = datetime.now()
    days = (today - datetime.strptime(start_date, "%Y-%m-%d")).days
    coin_id = coin.lower()
    data = cg_api.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days, interval='daily')  # Use the instance (cg_api) here
    new_data = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], unit='ms')
    new_data.set_index('timestamp', inplace=True)
    new_data = new_data[new_data.index >= start_datetime]  # Use start_datetime here for comparison
    new_data = new_data.rename(columns={'price': 'Close'})  # Rename the column before concatenating
    updated_price_data = pd.concat([price_data, new_data['Close']], axis=0)  # Use pandas.concat
    return updated_price_data




def chooseData(coin):
    data = pd.read_csv(f'data_csv/coin_{coin}.csv')
    price_data = data['Close']

    start_date = "2021-07-07"  # This is the first day not covered by the csv 
    updated_price_data = update_price_data(coin, price_data, start_date)

    price_data_array = updated_price_data.to_numpy()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    price_data_normalized = scaler.fit_transform(price_data_array.reshape(-1, 1))
    return scaler, price_data_normalized


def create_sequences(data, seq_length):
    inputs = []
    labels = []

    # New variable to store the last sequence
    last_sequence = []

    for i in range(len(data) - seq_length):
        inputs.append(data[i:i + seq_length, 0])
        labels.append(data[i + seq_length])

        if i == len(data) - seq_length - 1:
            last_sequence = data[i + 1:i + 1 + seq_length, 0]

    inputs = np.array(inputs).reshape(-1, seq_length, 1)
    labels = np.array(labels)

    return inputs, labels, np.array(last_sequence).reshape(1, -1, 1)


def sortData(data, train_ratio=0.7, val_ratio=0.2):
    seq_length = 30
    inputs, labels, last_sequence = create_sequences(data, seq_length)
    train_size = int(len(inputs) * train_ratio)
    val_size = int(len(inputs) * val_ratio)

    train_inputs = torch.tensor(inputs[:train_size], dtype=torch.float32)
    train_labels = torch.tensor(labels[:train_size], dtype=torch.float32)

    val_inputs = torch.tensor(inputs[train_size:train_size + val_size], dtype=torch.float32)
    val_labels = torch.tensor(labels[train_size:train_size + val_size], dtype=torch.float32)

    test_inputs = torch.tensor(inputs[train_size + val_size:], dtype=torch.float32)
    test_labels = torch.tensor(labels[train_size + val_size:], dtype=torch.float32)

    return train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels, last_sequence