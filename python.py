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
all_symbols_path = Path('Equities Universe - Symbols.csv')
all_symbols_data = pd.read_csv(all_symbols_path,
                               index_col='symbol', parse_dates=True, header=0)


# select symbol
symbol = all_symbols_data.index[0]  # get first symbol

# get file path
symbol_file_path = Path(symbol + '.csv')

# user start and end dates
start_date = '2025-01-02'
end_date = date.today().strftime('%Y-%m-%d')

# function for getting existing dates from all_symbols csv


def get_existing_dates(symbol):
    for chunk in pd.read_csv(all_symbols_path, index_col=0, chunksize=1):
        if symbol in chunk.index:
            existing_start_date = chunk.loc[symbol, 'start_date']
            existing_end_date = chunk.loc[symbol, 'end_date']
            return existing_start_date, existing_end_date


# get symbol csv file min max dates from all_symbols csv
if symbol_file_path.exists():
    # look up the min max dates in the all_symbols csv
    get_existing_dates(symbol)

else:
    # create data mask and save to new csv file
    date_range = pd.date_range(start=pd.to_datetime(
        start_date), end=pd.to_datetime(end_date))
    df_mask = pd.DataFrame(index=date_range)
    df_mask['Close'] = np.nan

    # save the mask to the symbol csv file
    df_mask.to_csv(symbol_file_path, index_label='Date')

    # log the start date and end date in the all_symbols csv
    all_symbols_data.loc[symbol, 'start_date'] = start_date
    all_symbols_data.loc[symbol, 'end_date'] = end_date
    all_symbols_data.to_csv(all_symbols_path, header=True)

    # look up the min max dates in the all_symbols csv
    get_existing_dates(symbol)

# get data from yfinance
# first define new data to get, excluding existing data based on existing dates


def new_data_date_range(symbol, start_date, end_date):
    date_ranges = []
    existing_start_date, existing_end_date = get_existing_dates(symbol)

    # case 0: no existing data
    if existing_start_date is None or existing_end_date is None:
        date_ranges.append((start_date, end_date + pd.Timedelta(days=1)))
        return date_ranges

    # case 1: start date is before existing range
    if start_date < existing_start_date and end_date < existing_end_date:
        date_ranges.append((start_date, end_date + pd.Timedelta(days=1)))
        return date_ranges

    # case 2: start date and end date are within existing range
    elif start_date >= existing_start_date and end_date <= existing_end_date:
        return date_ranges

        # case 3: end date is after existing range
    elif start_date >= existing_start_date and end_date > existing_end_date:
        start_date = existing_end_date + pd.Timedelta(days=1)
        date_ranges.append((start_date, end_date))
        return date_ranges

    # case 4: start date and end date are after existing range
    elif start_date > existing_end_date and end_date > existing_end_date:
        start_date = existing_end_date + pd.Timedelta(days=1)
        date_ranges.append((start_date, end_date))
        return date_ranges

    # case 5: start date is before existing range and end date is after existing range
    elif start_date < existing_start_date and end_date > existing_end_date:
        date_ranges.append(
            (start_date, existing_start_date + pd.Timedelta(days=1)))
        date_ranges.append(
            (existing_end_date + pd.Timedelta(days=1), end_date + pd.Timedelta(days=1)))

    return date_ranges


try:
    sart_date, end_date = new_data_date_range(symbol, start_date, end_date)
    print(f'start date: {start_date}\nend date: {end_date}')
except Exception:
    pass

# set parameters for getting data from yfinance
yfinance_params = {
    'start': start_date,  # [inclusive]
    'end': end_date,  # [exclusive]
    'auto_adjust': True,
    'rounding': True,
    'group_by': "Symbol"
}

# get new data from yfinance
yfinance_data = yf.download(symbol, **yfinance_params)

# # if the new data is latest, then append, else overwrite entire file


# # clean yfinance data
# # flatten multi-index columns
# yfinance_data.columns = [col[1] for col in yfinance_data.columns]

# # save Close as csv
# yfinance_data['Close'].to_csv(symbol_file_path)  # [Date, Close]
