import pandas as pd
from pathlib import Path
from get_data import get_symbol_data, symbol_data_to_returns_df
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import numpy as np
from charts import timeline, returns_heatmap, histogram
from stats import symbols_stats, symbols_and_results_stats



#pd.set_option('display.max_rows', None)
START_DATE = '2022-01-01'
end_date = '2023-07-10'

# get list of symbols
all_symbols_path = Path('Equities Universe - Symbols.csv')
all_symbols_data = pd.read_csv(all_symbols_path,index_col='symbol',parse_dates=['start_date', 'end_date'])
symbol = ['AAPL']

data = get_symbol_data(symbol, start_date=START_DATE, end_date=end_date)

# returns_df = symbol_data_to_returns_df(portfolio_1=data, market_index=None, start_date=START_DATE, end_date=end_date, value='relative_change_b')

returns_df = (
    data
    .pipe(symbol_data_to_returns_df, value='change')
    #.pipe(returns_stats)
)

# test: update symbols_stats to handle results_df and returns same info.  
stats_df = symbols_and_results_stats(data)
print(stats_df)
#results_stats_df = symbols_and_results_stats(returns_df, label='AAPL')
print(returns_df.iloc[-1,0])
#print(results_stats_df)
returns_heatmap(returns_df)
histogram(returns_df)
histogram(data)


# create histogram to hand symbols_df and results_df
