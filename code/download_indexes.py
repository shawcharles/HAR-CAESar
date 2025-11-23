import os
import pickle
import pandas as pd
import yfinance as yf

# Ensure data directory exists
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(DATA_DIR, exist_ok=True)

# Tickers: indices and ETF proxy for EM
TICKERS = {
    "SP500": "^GSPC",       # S&P 500 Index
    "FTSE100": "^FTSE",     # FTSE 100 Index
    "Nikkei225": "^N225",   # Nikkei 225 Index
    "MSCI_EM": "EEM",       # iShares MSCI Emerging Markets ETF as proxy
}

# Date range (can be adjusted as needed)
START_DATE = "2000-01-01"
END_DATE = None  # up to latest available

print("Downloading data from Yahoo Finance...")

series_list = []
for name, ticker in TICKERS.items():
    print(f"  - {name} ({ticker})")
    data = yf.download(ticker, start=START_DATE, end=END_DATE,
                       progress=False, auto_adjust=False)
    if data.empty:
        raise RuntimeError(f"No data returned for ticker {ticker} ({name})")
    # Prefer Adjusted Close if available, otherwise fall back to Close
    if "Adj Close" in data.columns:
        s = data["Adj Close"].copy()
    elif "Close" in data.columns:
        s = data["Close"].copy()
    else:
        raise RuntimeError(f"Ticker {ticker} ({name}) has neither 'Adj Close' nor 'Close' column")
    s.name = name
    series_list.append(s)

# Align on common business days and drop rows with any missing prices
prices = pd.concat(series_list, axis=1).dropna()

output_path = os.path.join(DATA_DIR, "indexes.pickle")
with open(output_path, "wb") as f:
    pickle.dump(prices, f)

print("Saved index prices to:", output_path)
print("Shape:", prices.shape)
print("Columns:", list(prices.columns))
print("Date range:", prices.index.min(), "to", prices.index.max())
