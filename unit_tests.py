import pandas as pd
from pathlib import Path
from get_data import get_symbol_data
from symbols_df_to_returns_df import time_series_to_returns_df
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import numpy as np
from charts import timeline, returns_heatmap, histogram
from stats import symbols_stats, symbols_and_results_stats
import datetime as dt



#pd.set_option('display.max_rows', None)
end_date = '2023-07-10'
START_DATE = dt.datetime.strptime(end_date, '%Y-%m-%d') - dt.timedelta(days=90)
print(START_DATE)

# get list of symbols
all_symbols_path = Path('Equities Universe - Symbols.csv')
all_symbols_data = pd.read_csv(all_symbols_path,index_col='symbol',parse_dates=['start_date', 'end_date'])
symbol = ['AAPL','TSLA', 'VOOG']

print(all_symbols_data.index[:30])

test_of_all_symbols = all_symbols_data.index[:30]

data = get_symbol_data(test_of_all_symbols, start_date=START_DATE, end_date=end_date)

returns_df = time_series_to_returns_df(time_series_1=data,time_series_2=None, value='relative_change_sign')
print(returns_df.head())
#