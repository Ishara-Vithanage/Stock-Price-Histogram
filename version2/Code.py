import yfinance as yf
import matplotlib.pyplot as plt

# Fetch historical data for Bitcoin (BTC-USD)
crypto_ticker = 'BTC-USD'
data = yf.download(crypto_ticker, start='2022-01-01', end='2023-01-01')

# Plot a histogram of the closing prices
plt.hist(data['Close'], bins=30)
plt.title('Bitcoin Price Distribution (BTC-USD)')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.show()
