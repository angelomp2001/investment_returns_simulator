# libraries
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import date
import numpy as np
# trying to use bigframes
# import bigframes.bigquery
# import bigframes.pandas as bpd
# bpd.options.display.progress_bar = None
# df = bpd.read_gbq("gs://'investment returns simulator'/symbols_universe.csv")

# get inputs
# get symbols universe
all_symbols_data = pd.read_csv('Equities Universe - Symbols.csv',
                               index_col='symbol', parse_dates=True, header=0)

all_symbols_path = Path('Equities Universe - Symbols.csv')
# select symbol
symbol = all_symbols_data.index[0]  # get first symbol

# get file path
symbol_file_path = Path(symbol + '.csv')

# user start and end dates
start_date = '2025-01-02'
end_date = date.today().strftime('%Y-%m-%d')


# get symbol csv file min max dates from all_symbols csv
if symbol_file_path.exists():
    # look up the min max dates in the all_symbols csv
    for chunk in pd.read_csv(all_symbols_path, index_col=0, chunksize=1):
        if symbol in chunk.index:
            existing_date_min = chunk.loc[symbol, 'start_date']
            existing_date_max = chunk.loc[symbol, 'end_date']
            break
else:
    # create symbol csv file and populate with data mask
    date_range = pd.date_range(start=pd.to_datetime(
        start_date), end=pd.to_datetime(end_date))
    df_mask = pd.DataFrame(index=date_range)
    df_mask['Close'] = np.nan
    df_mask.to_csv(symbol_file_path, index_label='Date')

    # save the mask to the symbol csv file
    df_mask.to_csv(symbol_file_path)

    # get data from yfinance
    yfinance_data = yf.download(
        symbol, group_by="Symbol", start=start_date, end=end_date, auto_adjust=True, rounding=True)

    # save user start date and end date to change later
    existing_date_min = start_date
    existing_date_max = end_date

    # log the start date and end date in the all_symbols csv
    all_symbols_data.loc[symbol, 'start_date'] = existing_date_min
    all_symbols_data.loc[symbol, 'end_date'] = existing_date_max
    all_symbols_data.to_csv(all_symbols_path, header=True)


print(
    f"Existing date range for {symbol}: {existing_date_min} to {existing_date_max}")
# current solution:
# existing_date_min = pd.read_csv(symbol_file_path, index_col='Date', nrows=1, parse_dates=True).index.min()
# existing_date_max = pd.read_csv(symbol_file_path, index_col='Date', parse_dates=True).index.max()

# actual start and end dates
# start_date = min(start_date, existing_date_min.strftime('%Y-%m-%d'))
# end_date = max(end_date, existing_date_max.strftime('%Y-%m-%d'))


# # clean data
# # flatten multi-index columns
# yfinance_data.columns = [col[1] for col in yfinance_data.columns]

# # save Close as csv
# yfinance_data['Close'].to_csv(symbol_file_path)  # [Date, Close]
