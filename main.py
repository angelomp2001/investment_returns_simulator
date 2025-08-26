from pathlib import Path
import pandas as pd
from datetime import date
from symbols_df_to_returns_df import time_series_to_returns_df
from get_data_chatgpt import get_symbol_data
from charts import returns_heatmap, histogram
from stats import symbols_and_results_stats

#data
all_symbols_path = Path('Equities Universe - Symbols.csv')
all_symbols_data = pd.read_csv(all_symbols_path,index_col='symbol',parse_dates=['start_date', 'end_date'])
all_symbols = all_symbols_data.index


#Parameters
portfolio_1_symbols = ['CRCL' ,'TSLA', 'NVDA','META','AMZN','GOOG','MSFT','O','AAPL']
market_index_symbols = ['VOOG'] #['VOOG','VGT','VTI','SPY','QQQ','VOO']
start_date = '2020-01-01'
end_date = pd.to_datetime("today")-pd.Timedelta(days=1)
end_date = end_date.normalize()
if end_date.weekday() >= 5:  # Saturday=5, Sunday=6
    end_date = end_date - pd.tseries.offsets.BDay(1)
else:
    end_date = end_date - pd.tseries.offsets.BDay(1)

end_date = end_date.strftime("%Y-%m-%d")

t = 60 - 20 # same as spreadsheet

# 1. Get data
portfolio_1 = get_symbol_data(all_symbols, start_date=start_date, end_date=end_date)
market_index = get_symbol_data('VOOG', start_date=start_date, end_date=end_date)

# 2. Get descriptive stats
# symbols_and_results_stats(portfolio_1)


# 3. Transform data into all possible returns df
returns_df = time_series_to_returns_df(time_series_1=portfolio_1, value='relative_change_sign')

# 4. Visualize Transformation
# returns_heatmap(returns_df)

# Get all possible returns df stats
stats_df = symbols_and_results_stats(returns_df)
print(stats_df)
