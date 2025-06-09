# libraries
import pandas as pd
import yfinance as yf

# trying to use bigframes
# import bigframes.bigquery
# import bigframes.pandas as bpd
# bpd.options.display.progress_bar = None
# df = bpd.read_gbq("gs://'investment returns simulator'/symbols_universe.csv")

# get inputs (symbols)
symbols = pd.read_csv('Equities Universe - Symbols.csv',
                      # series
                      index_col=0, parse_dates=True, names=['Symbol']).reset_index(drop=False)['Symbol']


yfinance_data = yf.download(symbols[1], group_by="Symbol",
                            start='2025-01-02', end='2025-01-03', auto_adjust=True, rounding=True)

# clean data
# flatten multi-index columns
yfinance_data.columns = [col[1] for col in yfinance_data.columns]

# save Close as csv
yfinance_data['Close'].to_csv(symbols[1] + '.csv')  # [Date, Close]
