from pathlib import Path
import pandas as pd
from datetime import date
from get_data import get_data

#Parameters
symbols = ['TSLA','NVDA','META','AMZN','GOOG','MSFT','O','AAPL']
start_date = '2020-01-01'
end_date = date.today().strftime('%Y-%m-%d')

get_data(symbols, start_date, end_date)
