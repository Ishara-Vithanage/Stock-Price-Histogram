import ccxt  # Import the ccxt library to fetch cryptocurrency data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def fetch_crypto_data(symbol, exchange_name, timeframe='1d', limit=30):
    # Fetch historical cryptocurrency price data and save to CSV.
    exchange = getattr(ccxt, exchange_name)()
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    data = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.to_csv(f'{symbol.replace("/", "_")}_data.csv', index=False)
    print(f"Data saved to {symbol.replace('/', '_')}_data.csv")
    return data


def load_data_from_csv(file_name):
    # Load cryptocurrency data from CSV file.
    data = pd.read_csv(file_name)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data


def calculate_volatility(prices):
    # Calculate volatility as the standard deviation of returns using numpy.
    returns = np.diff(prices) / prices[:-1]
    volatility = np.std(returns)
    return volatility


def visualize_trends(data, symbol):
    # Visualize price trends with dates on the x-axis and smaller fonts.
    plt.figure(figsize=(12, 6))
    plt.plot(data['timestamp'], data['close'], label=f'{symbol} Price', color='blue')
    plt.title(f'{symbol} Price Trend', fontsize=12)
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Price', fontsize=10)
    plt.xticks(fontsize=8, rotation=45)
    plt.yticks(fontsize=8)
    plt.legend(fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.show()


def main():
    # Define parameters
    symbol = input("Enter Cryptocurrency Pair (ex: BTC/USDT, ETH/BTC ): ")
    symbol = symbol.upper()  # Bitcoin to Tether
    exchange_name = 'binance'  # Binance exchange

    # Fetch data and save to CSV
    try:
        fetch_crypto_data(symbol, exchange_name)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Load data from CSV
    file_name = f'{symbol.replace("/", "_")}_data.csv'
    data = load_data_from_csv(file_name)

    # Calculate volatility
    volatility = calculate_volatility(data['close'].values)
    print(f'{symbol} Volatility (Std. Dev of Returns): {volatility:.5f}')

    # Visualize price trend
    visualize_trends(data, symbol)


main()