# libraries
import pandas as pd
import yfinance as yf
from datetime import date
from pathlib import Path

# trying to use bigframes
# import bigframes.bigquery
# import bigframes.pandas as bpd
# bpd.options.display.progress_bar = None
# df = bpd.read_gbq("gs://'investment returns simulator'/symbols_universe.csv")

# get inputs
# symbols
symbols = pd.read_csv('Equities Universe - Symbols.csv',
                      # series
                      # ['Symbol']
                      index_col=0, parse_dates=True, header=0)

# date range
start_date = '2025-01-01'
end_date = '2025-01-05'  # date.today().strftime('%Y-%m-%d')
symbol = symbols.index[0]

# save data location
csv_file_path = Path(f"{symbol}.csv")

# get data from yfinance
yfinance_data = yf.download(symbol, group_by="Symbol",
                            start=start_date, end=end_date, auto_adjust=True, rounding=True)

print("raw yfinance_data:\n", yfinance_data.head())

# clean data
# flatten multi-index columns
yfinance_data.columns = [col[1] for col in yfinance_data.columns]
print("flattened yfinance_data:\n", yfinance_data.head())

# save Close to csv
# check if csv file exists
if not csv_file_path.exists():
    yfinance_data.to_csv(csv_file_path)
else:
    existing_dates_min = pd.read_csv(
        csv_file_path, index_col='Date', parse_dates=True, nrows=1).index[0]
    print(
        f"existing_dates_min: {existing_dates_min}")
    num_rows = sum(1 for line in open(csv_file_path)) - \
        1  # Subtract 1 to account for the header
    print("num_rows:", num_rows)
    existing_dates_max = pd.read_csv(
        csv_file_path, index_col='Date', parse_dates=True, nrows=1, skiprows=num_rows-1).index[0]
    print(
        f"existing_dates_max: {existing_dates_max}")

#     existing_dates_max =  # the actual existing_dates_min
#     for existing_dates_chunk in pd.read_csv(csv_file_path, index_col='Date', parse_dates=True, chunksize=chunk_size).index:
#         if existing_dates_chunk.min() < existing_dates_min:
#             existing_dates_min = existing_dates_chunk.min()
#             existing_dates_max = existing_dates_chunk.max()

#         if existing_dates_chunk.max() > existing_dates_max:
#             existing_dates_max = existing_dates_chunk.max()
#         # chunks are whole rows

#     # Iterate over new data in chunks
#     for new_date, new_close in yfinance_data['Close'].iteritems():
#         # Check if the new date is not in existing dates
#         for old_date in existing_dates:
#             if new_date not in existing_dates:
#                 # make new df to then add to csv
#                 new_df = pd.DataFrame({'Close': [new_close]}, index=[new_date])

#                 # Append new data to the CSV file while avoiding duplicate headers
#                 new_df.to_csv(csv_file_path, mode='a',
#                               header=False, index=False)


# print(pd.read_csv(csv_file_path, index_col='Date', parse_dates=True).head())
