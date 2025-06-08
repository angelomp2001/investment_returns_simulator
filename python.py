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


# data = yf.download('AAPL', group_by="Symbol",
#                    start='2025-01-02', end='2025-01-03', auto_adjust=True)

# save to csv
# data.to_csv('AAPL.csv')
# aapl = pd.DataFrame(data['AAPL'])

print(pd.read_csv('AAPL.csv', index_col=0, parse_dates=True))

#                          AAPL             AAPL.1              AAPL.2              AAPL.3    AAPL.4
# Ticker
# Price                    Open               High                 Low               Close    Volume
# Date                      NaN                NaN                 NaN                 NaN       NaN
# 2025-01-02  248.3309608077005  248.5005651105234  241.23808511721737  243.26319885253906  55740700
