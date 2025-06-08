# get data
# download and save historical data from yahoo finance
import pandas as pd
import yfinance as yf

# trying to use bigframes
# import bigframes.bigquery
# import bigframes.pandas as bpd
# bpd.options.display.progress_bar = None
# df = bpd.read_gbq("gs://'investment returns simulator'/symbols_universe.csv")


df = pd.read_csv('Equities Universe - Symbols.csv',
                 index_col=0, parse_dates=True)
print(df.index)
