# get data
# download and save historical data from yahoo finance
import pandas as pd
import yfinance as yf

df = pd.read_csv('Equities Universe - Symbols.csv',
                 index_col=0, parse_dates=True)

print(df.head())
