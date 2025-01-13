import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fetch_data(ticker, start_date, end_date):
    # Get historical stock price data.
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        print("No data found for the given ticker and date range. Please try again.")
        return None
    data.reset_index(inplace=True)
    file_name = f"{ticker}.csv"
    data.to_csv(file_name, index=False)
    return file_name

def preprocess_data(file_name):
    # Preprocess the data using pandas for data management and numpy for calculations.
    try:
        data = pd.read_csv(file_name) # Read the CSV file into a DataFrame
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_name}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Could not parse file '{file_name}'. Check the file format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    # Fill missing values
    data.ffill(inplace=True) # Forward fill
    data.bfill(inplace=True) # Backward fill

    # Ensure 'Close' is numeric
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

    # Drop rows with NaN values after conversion
    data.dropna(subset=['Close'], inplace=True)

    # Convert 'Close' to a NumPy array for calculations
    close_prices = data['Close'].values

    # Calculate daily returns using numpy
    daily_returns = np.diff(close_prices) / close_prices[:-1]
    data['Daily Return'] = np.insert(daily_returns, 0, np.nan)  # Align with DataFrame length

    # Calculate moving averages using numpy
    def moving_average(values, window):
        return np.convolve(values, np.ones(window), 'valid') / window

    sma_50 = moving_average(close_prices, 50)
    sma_200 = moving_average(close_prices, 200)

    # Align moving averages with DataFrame length
    data['SMA_50'] = np.concatenate((np.full(49, np.nan), sma_50))
    data['SMA_200'] = np.concatenate((np.full(199, np.nan), sma_200))

    # Calculate rolling volatility (standard deviation) using numpy
    def rolling_std(values, window):
        return np.array([np.std(values[i:i + window]) for i in range(len(values) - window + 1)])

    volatility = rolling_std(daily_returns, 20)
    data['Volatility'] = np.concatenate((np.full(20, np.nan), volatility))

    return data

def analyze_data(data):
    # Data Visualization
    plt.figure(figsize=(14, 10))

    # Plot stock price and moving averages
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['Close'], label='Close Price', color='blue', linewidth=1.5)
    plt.plot(data.index, data['SMA_50'], label='50-Day MA', color='red', linestyle='--')
    plt.plot(data.index, data['SMA_200'], label='200-Day MA', color='green', linestyle='--')
    plt.title('Stock Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Plot Volatility
    plt.subplot(2, 1, 2)
    plt.plot(data.index, data['Volatility'], label='Volatility', color='purple', linewidth=1.5)
    plt.title('Stock Price Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    # Define the ticker symbol and period for stock data
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")
    ticker = input("Enter the ticker symbol (e.g., AAPL, MSFT, GOOGL): ")

    # Fetch and process data
    data_file = fetch_data(ticker, start_date, end_date)
    if data_file is not None:
        data = preprocess_data(data_file)
        if data is not None:
            analyze_data(data)

# Execute the program
if __name__ == "__main__":
    main()
