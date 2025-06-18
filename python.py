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
symbols = ['APLD','ANVS'] # get symbol(s)
print(f'symbols: {symbols}')

# user start and end dates
start_date = '2025-04-01'
end_date = '2025-05-01'  # date.today().strftime('%Y-%m-%d')

# immediately convert to datetime
start_date = pd.to_datetime(start_date)  # type: timestamp
end_date = pd.to_datetime(end_date)  # type: timestamp

# function for getting existing dates from all_symbols csv





def get_yfinance_data(symbols, start_date, end_date):
    print(f'Getting yfinance data for symbols: {symbols} from {start_date} to {end_date}')
     # get file path
    for symbol in symbols:
        print(f'Processing symbol: {symbol}')
        symbol_file_path = Path(symbol + '.csv')

        # check what you already have
        date_ranges = []
        case = np.nan
        #existing_start_date, existing_end_date = get_existing_dates(symbol)
        #def get_existing_dates(symbol):
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
                    #return existing_start_date, existing_end_date
        else:
            print(f'no existing dates')
            existing_start_date = None
            existing_end_date = None
        # create data mask and save to new csv file

        print(
            f'existing data for {symbol}:, date range:{existing_start_date}, {existing_end_date}')

        # case 0: no existing data
        if existing_start_date is None or existing_end_date is None:
            new_data_end_date = end_date
            new_data_start_date = start_date  
            date_ranges.append((new_data_start_date,new_data_end_date)) # [s, e]
            case = 0
            print(
                f'no existing data:  Case: {case}\n symbol:{symbol} \n new date range: {date_ranges[0][0]}, {date_ranges[0][1]}')

        # case 1: start date and end date are before existing range s---e___es===ee
        elif start_date < existing_start_date and end_date < existing_start_date:
            new_data_end_date = existing_start_date
            new_data_start_date = start_date 
            date_ranges.append((new_data_start_date, new_data_end_date)) # [s, es)
            case = 1
            print(
                f'no existing data:  Case: {case}\n new date range: {symbol}, {date_ranges[0][0]}, {date_ranges[0][1]}')

        # case 2: start date is before existing range s---es===e===ee
        elif start_date < existing_start_date and end_date <= existing_end_date:
            new_data_end_date = existing_start_date
            new_data_start_date = start_date
            date_ranges.append((new_data_start_date, new_data_end_date))  # [s, es)
            case = 2
            print(
                f'partial existing data:  Case: {case}\n new date range: {symbol}, {date_ranges[0][0]}, {date_ranges[0][1]}')

        # case 3: start date and end date are within existing range es===s===e===ee
        elif start_date >= existing_start_date and end_date <= existing_end_date:
            case = 3
            print(
                f'all data already existing:  Case: {case}\n date range:{existing_start_date}, {existing_end_date}')

            # case 4: end date is after existing range es===s===ee---e
        elif start_date >= existing_start_date and end_date > existing_end_date:
            new_data_start_date = existing_end_date + pd.Timedelta(days=1)
            new_data_end_date = end_date
            date_ranges.append((new_data_start_date, new_data_end_date)) # [s, e]
            case = 4
            print(
                f'partial existing data:  Case: {case}\n new date range: {symbol}, {date_ranges[0][0]}, {date_ranges[0][1]}')

        # case 5: start date and end date are after existing range es===ee___s---e
        elif start_date > existing_end_date and end_date > existing_end_date:
            new_data_start_date = existing_end_date + pd.Timedelta(days=1)
            new_data_end_date = end_date
            date_ranges.append((new_data_start_date, new_data_end_date)) # [s, e]
            case = 5
            print(
                f'no existing data:  Case: {case}\n new date range: {symbol}, {date_ranges[0][0]}, {date_ranges[0][1]}')

        # case 6: start date is before existing range and end date is after existing range s---es===ee---e
        elif start_date < existing_start_date and end_date > existing_end_date:
            new_data_start_date = start_date
            new_data_end_date = end_date
            date_ranges.append(
                (new_data_start_date, existing_start_date)) # [s, e)
            date_ranges.append(
                (existing_end_date + pd.Timedelta(days=1), new_data_end_date))  # [s, e]
            case = 6
            print(
                f'partial existing data:  Case: {case}\n new date range: {symbol}, {date_ranges[0][0]}, {date_ranges[0][1]}')
            print(
                f'partial existing data:  Case: {case}\n new date range: {symbol}, {date_ranges[1][0]}, {date_ranges[1][1]}')

        try:
            for start_date, end_date in date_ranges:
                # make data mask for incoming data
                print(f'making data mask for {symbol} from {start_date} to {end_date}')
                date_range = pd.date_range(min(filter(pd.notnull, [start_date, existing_start_date])), max(filter(pd.notnull, [end_date, existing_end_date])))
                df_mask = pd.DataFrame(index=date_range)
                df_mask['Close'] = np.nan  # creates column with blanks in csv
                print(f'df_mask new data index range: {df_mask.index.min()} to {df_mask.index.max()}')
                try:
                    existing_symbol_data = pd.read_csv(symbol_file_path, index_col='Date', parse_dates=True)
                    df_mask.loc[existing_symbol_data.index, 'Close'] = existing_symbol_data['Close']
                    print(f'df_mask with existing data (if any) index range: {df_mask.index.min()} to {df_mask.index.max()}')
                except FileNotFoundError:
                    print(f'existing data not found for adding to data mask.')            
                
                # save data mask to csv
                df_mask.to_csv(symbol_file_path, index_label='Date')
                print(f'df_mask index range saved: {df_mask.index.min()} to {df_mask.index.max()}')

                # get data from yfinance
                yfinance_params = {
                    'start': start_date,  # [inclusive]
                    'end': end_date + pd.Timedelta(days=1),  # [exclusive] 
                    'auto_adjust': True,
                    'rounding': True,
                    'group_by': "Symbol"
                }
                print(f'getting data from yfinance for {symbol} from {start_date} to {end_date}')
                yfinance_data = yf.download(symbol, **yfinance_params)
                # clean yfinance data
                # flatten multi-index columns
                print(f'flattening yfinance data columns for {symbol}')
                yfinance_data.columns = [col[1] for col in yfinance_data.columns]
                #print(f'yfinance: \nindex: {yfinance_data.index.loc[:5]}\n head: {yfinance_data['Close'].head(5)}')

                # save Close as csv
                print(f'saving yfinance data for {symbol} to {symbol_file_path}')
                print(f'data mask is deleted upon saving :(')

                try:
                    # read existing data and update Close column
                    print(f'if existing data exists, updating existing data for {symbol} in {symbol_file_path}')
                    existing_symbol_data = pd.read_csv(symbol_file_path, index_col='Date', parse_dates=True)
                    # update existing data with new Close values
                    print(f'updating existing data for {symbol} in {symbol_file_path}')
                    print(f'existing data index range w/ data mask: {existing_symbol_data.index.min()} to {existing_symbol_data.index.max()}') 
                    existing_symbol_data.loc[yfinance_data.index, 'Close'] = yfinance_data['Close']
                    existing_symbol_data.to_csv(symbol_file_path, index_label='Date')
                    print(f'existing data index range: {existing_symbol_data.index.min()} to {existing_symbol_data.index.max()}')


                    
                except FileNotFoundError:
                    print(f'existing data not found for adding to data mask.')            
                    # save data mask to csv
                    yfinance_data['Close'].to_csv(symbol_file_path)  # [Date, Close]

                # log new data dates in all symbols csv
                # Assign the minimum date to the DataFrame
                print(f'updating all_symbols_data for {symbol}: \
                    \n new date range: {new_data_start_date}, {new_data_end_date}\
                    \n existing date range: {existing_start_date}, {existing_end_date}')
                all_symbols_data.loc[symbol, 'start_date'] = min(
                    filter(pd.notnull, [
                        pd.to_datetime(new_data_start_date),
                        pd.to_datetime(existing_start_date)]))

                # Assign the max date to the DataFrame
                all_symbols_data.loc[symbol, 'end_date'] = max(
                    filter(pd.notnull, [
                        pd.to_datetime(new_data_end_date, errors='coerce'), #off by one in case 6
                        pd.to_datetime(existing_end_date, errors='coerce')]))
                
                # save all_symbols_data to csv
                print(f'saving all_symbols_data to {all_symbols_path}')
                all_symbols_data.to_csv(all_symbols_path, index_label='symbol')
                print(f'new data saved for {symbol} from range: {all_symbols_data.loc[symbol, "start_date"]} to {all_symbols_data.loc[symbol, "end_date"]}')

                # update existing_start_date and existing_end_date for next iteration
                existing_start_date, existing_end_date = get_existing_dates(symbol)

        except Exception as e:
            print(f'Error getting data from yfinance: {e}')


get_yfinance_data(symbols, start_date, end_date)
