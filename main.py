from pathlib import Path
import pandas as pd
from datetime import date
from get_data import get_symbol_data
from charts import returns_heatmap, histogram
from get_data import symbol_data_to_returns_df
from stats import symbols_stats

#Parameters
portfolio_1_symbols = ['TSLA', 'NVDA','META']#,'AMZN','GOOG','MSFT','O','AAPL']
market_index_symbols = ['VOOG'] #['VOOG','VGT','VTI','SPY','QQQ','VOO']
start_date = '2024-01-01'
end_date = date.today().strftime('%Y-%m-%d')

# 1. Get data
portfolio_1 = get_symbol_data(portfolio_1_symbols, start_date=start_date, end_date=end_date)
market_index = get_symbol_data(market_index_symbols, start_date=start_date, end_date=end_date)

# 2. Get descriptive stats
symbols_stats(portfolio_1)

# 3. Transform data into all possible returns df
returns_df = symbol_data_to_returns_df(portfolio_1=portfolio_1, market_index=market_index, start_date=start_date, end_date=end_date, value='relative_change_b')

# 4. Visualize Transformation
returns_heatmap(returns_df)

# Get all possible returns df stats

