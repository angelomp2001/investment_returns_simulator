from pathlib import Path
import pandas as pd
from datetime import date
from get_data import get_symbol_data
from charts import two_d_heatmap
from stats import stats

#Parameters
portfolio_1 = ['TSLA'] #['TSLA','NVDA','META','AMZN','GOOG','MSFT','O','AAPL']
portfolio_2 = ['VOOG'] #['VOOG','VGT','VTI','SPY','QQQ','VOO']
start_date = '2020-01-01'
end_date = date.today().strftime('%Y-%m-%d')

df_1 = get_symbol_data(portfolio_1, start_date, end_date)
df_2 = get_symbol_data(portfolio_2, start_date, end_date)

portfolio = ['MSFT']
portfolio_df = get_symbol_data(portfolio, start_date, end_date)
portfolio_returns = symbol_data_to_returns_df(portfolio_df)
print(f'\nportfolio_df_1:\n{portfolio_returns.head()}')

market_index = ['VOOG']
market_index_df = get_symbol_data(market_index, start_date, end_date)
market_index_returns = symbol_data_to_returns_df(market_index_df)
print(f'\nportfolio_df_2:\n{market_index_returns.head()}')

net_returns = symbol_data_to_returns_df(portfolio_df, market_index_df)
print(f'\nnet_returns:\n{net_returns}')

stats(portfolio_returns, market_index_returns)

returns_heatmap(net_returns)
