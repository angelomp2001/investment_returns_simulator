from pathlib import Path
import pandas as pd
from datetime import date
from get_data import get_symbol_data
from charts import returns_heatmap
from get_data import symbol_data_to_returns_df
from stats import stats

#Parameters
portfolio_1 = ['TSLA'] #['TSLA','NVDA','META','AMZN','GOOG','MSFT','O','AAPL']
portfolio_2 = ['VOOG'] #['VOOG','VGT','VTI','SPY','QQQ','VOO']
start_date = '2020-01-01'
end_date = date.today().strftime('%Y-%m-%d')

df_1 = get_symbol_data(symbols=portfolio_1, start_date=start_date, end_date=end_date)
df_2 = get_symbol_data(symbols=portfolio_2, start_date=start_date, end_date=end_date)

portfolio = ['MSFT']
portfolio_df = get_symbol_data(symbols=portfolio, start_date=start_date, end_date=end_date)
portfolio_returns = symbol_data_to_returns_df(portfolio_1=portfolio_df, start_date=start_date, end_date=end_date)
print(f'\nportfolio_df_1:\n{portfolio_returns.head()}')

market_index = ['VOOG']
market_index_df = get_symbol_data(symbols=market_index, start_date=start_date, end_date=end_date)
market_index_returns = symbol_data_to_returns_df(portfolio_1=market_index_df, start_date=start_date, end_date=end_date)
print(f'\nportfolio_df_2:\n{market_index_returns.head()}')

net_returns = symbol_data_to_returns_df(portfolio_1=portfolio_df, market_index=market_index_df)
print(f'\nnet_returns:\n{net_returns}')

stats(symbol_df=portfolio_returns)

returns_heatmap(returns_df=net_returns)
