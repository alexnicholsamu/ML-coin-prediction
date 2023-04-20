from pycoingecko import CoinGeckoAPI as cg
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def update_price_data(coin, price_data, start_date):
    start_timestamp = datetime.strptime(start_date, "%Y-%m-%d").timestamp()
    today = datetime.now()
    days = (today - datetime.strptime(start_date, "%Y-%m-%d")).days
    coin_id = coin.lower()
    data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days, interval='daily')
    new_data = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], unit='ms')
    new_data.set_index('timestamp', inplace=True)
    new_data = new_data[new_data.index >= start_timestamp]
    updated_price_data = price_data.append(new_data.rename(columns={'price': 'Close'}))
    return updated_price_data['Close']


def chooseData(coin):
    data = pd.read_csv(f'data_csv/coin_{coin}.csv')
    price_data = data['Close']

    start_date = "2021-07-07"  # This is the first day not covered by the csv 
    updated_price_data = update_price_data(coin, price_data, start_date)

    price_data_array = updated_price_data.to_numpy()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    price_data_normalized = scaler.fit_transform(price_data_array.reshape(-1, 1))
    return scaler, price_data_normalized