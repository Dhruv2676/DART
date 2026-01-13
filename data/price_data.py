import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

with open('./tickers.txt', 'r') as f:
    tickers = f.read().split()
print(f"Got {len(tickers)} Tickers...")

def get_stock_ohlcv_df(ticker, start_date="2015-01-01", end_date="2024-01-01"):
  odf = yf.download(ticker, start_date, end_date, auto_adjust=True)
  rows = []
  for date, row in odf.iterrows():
    rows.append({'Date': date, 'Open': row['Open', ticker], 'High': row['High', ticker], 'Low': row['Low', ticker], 'Close': row['Close', ticker], 'Volume': int(row['Volume', ticker])})

  df = pd.DataFrame(rows, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
  df.set_index('Date', inplace=True)
  df = df.sort_values('Date')
  return df

def get_indicators(df):
  df.ta.macd(close=df['Close'], fast=12, slow=26, signal=9, append=True)
  df.ta.rsi(close=df['Close'], append=True)
  df.ta.bbands(close=df['Close'], append=True)
  df.ta.adx(high=df['High'], low=df['Low'], close=df['Close'], append=True)
  df.ta.obv(close=df['Close'], volume=df['Volume'], append=True)

  df.dropna(inplace=True)
  df.rename(columns={"MACD_12_26_9": "MACD", "MACDh_12_26_9": "MACD_Hist", "MACDs_12_26_9": "MACD_Signal", "RSI_14": "RSI",
                     "BBL_5_2.0_2.0": "BB_Lower", "BBM_5_2.0_2.0": "BB_Middle", "BBU_5_2.0_2.0": "BB_Upper", "BBB_5_2.0_2.0": "BB_Width",
                     "BBP_5_2.0_2.0": "BB_Percent", "ADX_14": "ADX", "ADXR_14_2": "ADXR", "DMP_14": "DI_Pos", "DMN_14": "DI_Neg",
                     "OBV": "OBV"}, inplace=True)
  return df

data_folder = "./price/"
returns_file = "./returns.csv"

returns = {}

for ticker in tickers:
    print(f"Getting OHLCV data for {ticker}...")
    stock_ohlcv = get_stock_ohlcv_df(ticker)

    time.sleep(1)
   
    print(f"Calculating Indicators for {ticker}...")
    final_stock_data = get_indicators(stock_ohlcv)
    final_stock_data.dropna(inplace=True)

    final_stock_data['Returns'] = np.log(final_stock_data['Close'] / final_stock_data['Close'].shift(1))
    final_stock_data.dropna(inplace=True)
    if final_stock_data.index.has_duplicates:
        num_duplicates = final_stock_data.index.duplicated().sum()
        final_stock_data = final_stock_data[~final_stock_data.index.duplicated(keep='first')]
    returns[ticker] = final_stock_data['Returns']
   
    print(f"We got {len(final_stock_data)} trading days data for {ticker}...")
    final_stock_data.to_csv(f'{data_folder}{ticker}.csv')
    print('\n')

print('\n')
print(f"All {len(tickers)} ticker's data stored in {data_folder} successfully...")
print('\n')

returns_df = pd.DataFrame(returns)
returns_df.to_csv(returns_file)
print(f"log returns stored at {returns_file} successfully...")

