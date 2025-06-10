# libraries
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import date
# trying to use bigframes
# import bigframes.bigquery
# import bigframes.pandas as bpd
# bpd.options.display.progress_bar = None
# df = bpd.read_gbq("gs://'investment returns simulator'/symbols_universe.csv")

# get inputs
# get symbols universe
all_symbols = pd.read_csv('Equities Universe - Symbols.csv',
                          index_col='Symbol', parse_dates=True, header=0)

# select symbol
symbol = all_symbols.index[0]  # get second symbol

# user start and end dates
start_date = '2025-01-02'
end_date = date.today().strftime('%Y-%m-%d')

# get file path
symbol_file_path = Path(symbol + '.csv')

# get symbol csv file min max dates
existing_date_min = pd.read_csv(
    symbol_file_path, index_col='Date', nrows=1, parse_dates=True).index.min()

existing_date_max = pd.read_csv(
    symbol_file_path, index_col='Date', parse_dates=True).index.max()

# actual start and end dates
start_date = min(start_date, existing_date_min.strftime('%Y-%m-%d'))
end_date = max(end_date, existing_date_max.strftime('%Y-%m-%d'))

# get data from yfinance
yfinance_data = yf.download(symbol, group_by="Symbol",
                            start=start_date, end=end_date, auto_adjust=True, rounding=True)

# clean data
# flatten multi-index columns
yfinance_data.columns = [col[1] for col in yfinance_data.columns]

# save Close as csv
yfinance_data['Close'].to_csv(symbol_file_path)  # [Date, Close]
