import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import date
import numpy as np
import datetime as dt
from tqdm import tqdm
import time


############################## Information on where data is and how to get it ##############################
all_symbols_path = Path('Equities Universe - Symbols.csv')
all_symbols_data = pd.read_csv(all_symbols_path,index_col='symbol',parse_dates=['start_date', 'end_date'])

chunksize = 10000  # Number of rows per chunk for reading large CSV files

def symbol_data_existing_dates(
        symbol: str = None
        ):
        '''
        Returns start and end date for existing data
        '''
        #initialize vars
        symbol = symbol if isinstance(symbol, str) else [symbol]
        symbol_file_path = Path('symbols/' + symbol + '.csv')
        existing_start_date = None
        existing_end_date = None
        
        if symbol_file_path.exists():
            df = pd.read_csv(all_symbols_path, index_col=0)
            existing_start_date = pd.to_datetime(df.loc[symbol, 'start_date'])
            existing_end_date = pd.to_datetime(df.loc[symbol, 'end_date'])

        # if symbol_file_path.exists():
        #     #print(f'file exists')
        #     for chunk in pd.read_csv(all_symbols_path, index_col=0):
        #         if symbol in chunk.index:
        #             existing_start_date = pd.to_datetime(chunk.loc[symbol, 'start_date'])
        #             existing_end_date = pd.to_datetime(chunk.loc[symbol, 'end_date'])
        else:
            print(f'dates not found')
            existing_start_date = None
            existing_end_date = None
        return existing_start_date, existing_end_date


def get_symbol_data(
        symbols: list[str] = None,
        start_date: date = None,
        end_date: date = None
        ):
    '''
    input: symbol, start date, end date.
    output: symbol.csv, index: start date - end date.  Symbol: price_close
    output: symbols_data_xxxx, index: start date - end date. cols: (symbols): price_close
    0. Initialize vars
    1. Check if symbol data already stored in db
    2. Define new data data range based on existing data date range
        case 0: no existing data
        case 1: start date and end date are before existing range: s---e___es===ee
        case 2: start date is before existing range: s---es===e===ee
        case 3: start date and end date are within existing range: es===s===e===ee
        case 4: start date and end date are after existing range: es===s===ee---e
        case 5: start date and end date are after existing range: es===ee___s---e
        case 6: start date is before existing range and end date is after existing range: s---es===ee---e
    3. Get new data
        make data mask for incoming data
        save data mask to csv
        get data from yfinance
        clean yfinance data
        save Close as csv
    4. Return data
    '''

    '0. Initialize vars'
    symbols = [symbols] if isinstance(symbols, str) else symbols
    start_date = pd.to_datetime(start_date)  
    end_date = pd.to_datetime(end_date) 
    case = np.nan
    df = pd.DataFrame()
    folder_name = "symbols_data"            # folder where you want to save your files
    folder_path = Path(folder_name)
    folder_path.mkdir(exist_ok=True)

    # check for existing symbols data df
    symbols_data_paths = list(Path('.').glob('symbols/symbols_data_*.csv'))
    for each_path in symbols_data_paths:
        each_symbols_data = pd.read_csv(each_path, index_col='Date', parse_dates=True)
        
        # Check if all symbols exist as columns in the DataFrame
        if set(symbols).issubset(each_symbols_data.columns):
            # Select the data for the specified date range and symbols
            subset_df = each_symbols_data.loc[start_date:end_date, symbols]
        
            # Check if the subset DataFrame is empty (no data in that date range)
            if subset_df.empty:
                continue
            else:
                df = subset_df
                return df

    # 1. Check if symbol data already stored in db
    for symbol in symbols:
        
        # initialize loop vars
        symbol_file_path = Path('symbols/' + symbol + '.csv')
        date_ranges = []
        existing_start_date, existing_end_date = symbol_data_existing_dates(symbol)

        if existing_start_date is None or existing_end_date is None:
            new_data_start_date = start_date
            new_data_end_date = end_date
            date_ranges.append((new_data_start_date, new_data_end_date))
            case = 0

        elif start_date < existing_start_date and end_date < existing_start_date:
            new_data_start_date = start_date
            new_data_end_date = existing_start_date
            date_ranges.append((new_data_start_date, new_data_end_date))
            case = 1

        elif start_date < existing_start_date and end_date <= existing_end_date:
            new_data_start_date = start_date
            new_data_end_date = existing_start_date
            date_ranges.append((new_data_start_date, new_data_end_date))
            case = 2

        elif start_date >= existing_start_date and end_date <= existing_end_date:
            case = 3

        elif start_date >= existing_start_date and end_date > existing_end_date:
            new_data_start_date = existing_end_date + pd.Timedelta(days=1)
            new_data_end_date = end_date
            date_ranges.append((new_data_start_date, new_data_end_date))
            case = 4

        elif start_date > existing_end_date and end_date > existing_end_date:
            new_data_start_date = existing_end_date + pd.Timedelta(days=1)
            new_data_end_date = end_date
            date_ranges.append((new_data_start_date, new_data_end_date))
            case = 5

        elif start_date < existing_start_date and end_date > existing_end_date:
            new_data_start_date = start_date
            new_data_end_date = end_date
            date_ranges.append((new_data_start_date, existing_start_date))
            date_ranges.append((existing_end_date + pd.Timedelta(days=1), new_data_end_date))
            case = 6

        # print(f'case: {case}')
        

        try:
            for start_date, end_date in tqdm(
                date_ranges,
                desc='Getting Data', 
                unit='symbol', 
                leave=True,
                colour='green',  
                ascii=(" ", "█"),
                ncols=100
            ):
                # make data mask for incoming data
                date_range = pd.date_range(
                    min(filter(pd.notnull, [start_date, existing_start_date])),
                    max(filter(pd.notnull, [end_date, existing_end_date]))
                )
                df_mask = pd.DataFrame(index=date_range)
                df_mask['Close'] = np.nan

                try:
                    existing_symbol_data = pd.read_csv(symbol_file_path, index_col='Date', parse_dates=True)
                    df_mask.loc[existing_symbol_data.index, 'Close'] = existing_symbol_data['Close']
                except FileNotFoundError:
                    print(f'existing data not found for {symbol}, creating new file.')

                # save data mask to csv
                df_mask.to_csv(symbol_file_path, index_label='Date')

                # get and format data from yfinance
                yfinance_params = {
                    'start': start_date,  # [default inclusive]
                    'end': end_date + pd.Timedelta(days=1),  # [default exclusive] 
                    'auto_adjust': True,
                    'rounding': True,
                    'group_by': "Symbol"
                }
                yfinance_data = yf.download(symbol, **yfinance_params)
                
                # flatten multi-index columns
                yfinance_data.columns = [col[1] for col in yfinance_data.columns]

                if yfinance_data.empty:
                    all_symbols_data = pd.read_csv(all_symbols_path,index_col='symbol',parse_dates=['start_date', 'end_date'])
                    all_symbols_data = all_symbols_data.drop(symbol, axis=0)
                    all_symbols_data.to_csv(all_symbols_path, index_label='symbol')
                    symbol_file_path.unlink()
                    print('yfinance returned no data. symbol/file deleted')
                    

                else:
                    # read existing data, update Close column, save as symbol.csv
                    existing_symbol_data = pd.read_csv(symbol_file_path, index_col='Date', parse_dates=True)
                    existing_symbol_data.loc[yfinance_data.index, 'Close'] = yfinance_data['Close']
                    existing_symbol_data.to_csv(symbol_file_path, index_label='Date')
                
                    # read existing data, update dates columns, save all_symbols.csv
                    all_symbols_data = pd.read_csv(all_symbols_path,index_col='symbol',parse_dates=['start_date', 'end_date'])
                    all_symbols_data.loc[symbol, 'start_date'] = min(filter(pd.notnull, [
                        pd.to_datetime(new_data_start_date),
                        pd.to_datetime(existing_start_date)]))
                    all_symbols_data.loc[symbol, 'end_date'] = max(filter(pd.notnull, [
                        pd.to_datetime(new_data_end_date, errors='coerce'),
                        pd.to_datetime(existing_end_date, errors='coerce')]))
                    all_symbols_data.to_csv(all_symbols_path, index_label='symbol')

                #re-get existing dates from now-saved file.  
                existing_start_date, existing_end_date = symbol_data_existing_dates(symbol)

        except Exception as e:
            print(f'Error getting data from yfinance for {symbol}: {e}')
            

        # Read and populate final data frame
         
        try:
            symbol_df = pd.read_csv(symbol_file_path, index_col='Date', parse_dates=True)
            
            # Ensure index is datetime and sorted
            symbol_df.index = pd.to_datetime(symbol_df.index)
            symbol_df = symbol_df.sort_index()
            
            df.loc[start_date:end_date, symbol] = symbol_df.loc[start_date:end_date, 'Close'].copy()
            
            # Filter to the desired range
            # filtered_df = symbol_df.loc[start_date:end_date, ['Close']].copy()
            # filtered_df.columns = [symbol]
           
            # # Align index with df (intersect only on dates that exist in both)
            # valid_dates = df.index.intersection(filtered_df.index)
            
            # # Use .loc for batch assignment
            # df.loc[valid_dates, symbol] = filtered_df.loc[valid_dates, symbol]

        except Exception as e:
            print(f"Error processing final data for {symbol}: {e}")

    df.to_csv(f'symbols_data/symbols_data_{time.time()}' + '.csv', index_label='Date')
    return df

def symbol_data_to_returns_df(
    portfolio_1: pd.DataFrame = None,
    market_index: pd.DataFrame = None,
    start_date: str = None,
    end_date: str = None,
    value: str = None
):
    """
    Computes a returns matrix from one or two DataFrames.
    If both portfolio_1 and market_index are provided, returns the difference in returns.
    Each cell (i,j) is the return from date i to date j.
    portfolio_1: the main portfolio
    market_index: the market index portfolio_1 is compared gainst, if provided.
    start_date: start date of comparison
    end_date: end date of comparison
    value: what is returned: 'close', 'change', 'relative_change', 'change_b', 'relative_change_b'
    """
    
    def compute_returns(df: pd.DataFrame, value: str, start_date, end_date):
        # dropna
        df = df.dropna(axis=0, how='all')

        # set start and end dates
        if start_date is None:
            start_date = df.index.min()
        else:
            start_date = pd.to_datetime(start_date)

        if end_date is None:
            end_date = df.index.max()
        else:
            end_date = pd.to_datetime(end_date)

        # initialize vars
        df = df.loc[start_date:end_date]
        date_index = df.index
        m = len(date_index)
        returns_matrix = np.zeros((m, m))

        for symbol in df.columns:
            #vectorize prices
            prices = df[symbol].to_numpy(dtype=float)

            # Generate 2D grids: prices[i, j] = (price at j) / (price at i)
            start_prices = prices[:, np.newaxis]  # Shape (n,1)
            end_prices = prices[np.newaxis, :]    # Shape (1,n)

            with np.errstate(divide='ignore', invalid='ignore'):
                if value == 'close':
                    # contrib will be the close price on that day
                    contrib = end_prices
                elif value == 'change':
                    # new var which represents previous col (end_date) price
                    prev_end_date_prices = np.roll(prices, 1)
                    # calculate returns
                    returns = (prices / prev_end_date_prices) - 1
                    returns[0] = 0  # No previous day for first row
                    # tile the returns 
                    contrib = np.tile(returns, (m, 1))
                elif value is None or value == 'relative_change':
                    # this is the default
                    contrib = (end_prices / start_prices) - 1
                elif value == 'change_b':
                    # new var which represents previous col (end_date) price
                    prev_end_date_prices = np.roll(prices, 1)
                    # calculate returns as binary (1= True, 0 <= False)
                    returns = ((prices / prev_end_date_prices) - 1) > 0
                    # first value does not exist
                    returns[0] = False
                    # replace True with 1, False with -1 
                    contrib = np.where(np.tile(returns, (m, 1)), 1, -1)
                elif value == 'relative_change_b':
                    contrib = np.where((end_prices / start_prices) - 1 > 0, 1, -1)
                else:
                    contrib = np.zeros((m, m))

            # replace nan with 0
            contrib = np.nan_to_num(contrib)
            # Only upper triangle is valid (start < end)
            mask = np.triu(np.ones((m, m), dtype=bool), k=1)
            returns_matrix[mask] += contrib[mask]

        return pd.DataFrame(
            returns_matrix,
            index=date_index,
            columns=date_index
        )

    # Compute returns for each provided DataFrame
    returns_1 = compute_returns(portfolio_1, value, start_date, end_date) if portfolio_1 is not None else None
    returns_2 = compute_returns(market_index, value, start_date, end_date) if market_index is not None else None

    if returns_1 is not None and returns_2 is not None:
        return returns_1.subtract(returns_2, fill_value=0)

    if returns_1 is not None:
        return returns_1
    elif returns_2 is not None:
        return returns_2
    else:
        return None

    

    
