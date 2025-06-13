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
all_symbols_data = pd.read_csv(
    all_symbols_path,
    index_col='symbol',
    # Explicitly specify columns to parse as dates
    parse_dates=['start_date', 'end_date']
)


# select symbol
symbol = all_symbols_data.index[0]  # get first symbol

# get file path
symbol_file_path = Path(symbol + '.csv')

# user start and end dates
start_date = '2025-01-02'
end_date = '2025-01-03'  # date.today().strftime('%Y-%m-%d')

# immediately convert to datetime
start_date = pd.to_datetime(start_date)  # type: timestamp
end_date = pd.to_datetime(end_date)  # type: timestamp

# function for getting existing dates from all_symbols csv


def get_existing_dates(symbol):
    if symbol_file_path.exists():
        print(f'file exists')
        for chunk in pd.read_csv(all_symbols_path, index_col=0, chunksize=1):
            if symbol in chunk.index:
                existing_start_date = pd.to_datetime(
                    chunk.loc[symbol, 'start_date'])
                existing_end_date = pd.to_datetime(
                    chunk.loc[symbol, 'end_date'])
                print(
                    f'get_existing_dates(): existing_start_date: {existing_start_date}, existing_end_date: {existing_end_date}')
                return existing_start_date, existing_end_date
    else:
        print(f'no existing dates')
        return None, None
        # create data mask and save to new csv file


def get_yfinance_data(symbol, start_date, end_date):
    # check what you already have
    date_ranges = []
    case = 0
    existing_start_date, existing_end_date = get_existing_dates(symbol)
    print(
        f'existing data: {symbol}:, date range:{existing_start_date}, {existing_end_date}')

    # case 0: no existing data
    if existing_start_date is None or existing_end_date is None:
        date_ranges.append((start_date, end_date + pd.Timedelta(days=1)))
        print(
            f'no existing data:  Case: {case}\n new date range: {symbol}')

    # case 1: start date is before existing range
    elif start_date < existing_start_date and end_date < existing_end_date:
        date_ranges.append((start_date, end_date + pd.Timedelta(days=1)))
        case = 1
        print(
            f'partial existing data:  Case: {case}\n new date range: {symbol}, {date_ranges[0]}, {date_ranges[1]}')

    # case 2: start date and end date are within existing range
    elif start_date >= existing_start_date and end_date <= existing_end_date:
        case = 2
        print(
            f'all data already existing:  Case: {case}\n date range:{existing_start_date}, {existing_end_date}')

        # case 3: end date is after existing range
    elif start_date >= existing_start_date and end_date > existing_end_date:
        start_date = existing_end_date + pd.Timedelta(days=1)
        date_ranges.append((start_date, end_date))
        case = 3
        print(
            f'partial existing data:  Case: {case}\n new date range: {symbol}, {date_ranges[0]}, {date_ranges[1]}')

    # case 4: start date and end date are after existing range
    elif start_date > existing_end_date and end_date > existing_end_date:
        start_date = existing_end_date + pd.Timedelta(days=1)
        date_ranges.append((start_date, end_date))
        case = 4
        print(
            f'no existing data:  Case: {case}\n new date range: {symbol}, {date_ranges[0]}, {date_ranges[1]}')

    # case 5: start date is before existing range and end date is after existing range
    elif start_date < existing_start_date and end_date > existing_end_date:
        date_ranges.append(
            (start_date, existing_start_date + pd.Timedelta(days=1)))
        date_ranges.append(
            (existing_end_date + pd.Timedelta(days=1), end_date + pd.Timedelta(days=1)))
        case = 5
        print(
            f'partial existing data:  Case: {case}\n new date range: {symbol}, {date_ranges[0]}, {date_ranges[1]}')
        print(
            f'partial existing data:  Case: {case}\n new date range: {symbol}, {date_ranges[2]}, {date_ranges[3]}')

    try:
        for start_date, end_date in date_ranges:
            # make data mask for incoming data
            date_range = pd.date_range(start=pd.to_datetime(
                start_date), end=pd.to_datetime(end_date))
            df_mask = pd.DataFrame(index=date_range)
            df_mask['Close'] = np.nan  # creates column with blanks in csv
            df_mask.to_csv(symbol_file_path, index_label='Date')

            # get data from yfinance
            yfinance_params = {
                'start': start_date,  # [inclusive]
                'end': end_date,  # [exclusive]
                'auto_adjust': True,
                'rounding': True,
                'group_by': "Symbol"
            }
            yfinance_data = yf.download(symbol, **yfinance_params)
            # clean yfinance data
            # flatten multi-index columns
            yfinance_data.columns = [col[1] for col in yfinance_data.columns]

            # save Close as csv
            yfinance_data['Close'].to_csv(symbol_file_path)  # [Date, Close]

            # log new data dates
            all_symbols_data.loc[symbol, 'start_date'] = pd.to_datetime(
                start_date, format='%Y-%m-%d')
            all_symbols_data.loc[symbol, 'end_date'] = pd.to_datetime(
                end_date, format='%Y-%m-%d')
            all_symbols_data.to_csv(all_symbols_path, header=True)
    except Exception as e:
        print(f'Error getting data from yfinance: {e}')


get_yfinance_data(symbol, start_date, end_date)
