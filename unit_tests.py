import pandas as pd
from pathlib import Path
from get_data import get_symbol_data, symbol_data_to_returns_df
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import numpy as np
from charts import timeline, returns_heatmap, histogram
from stats import stats 


#pd.set_option('display.max_rows', None)
START_DATE = '2020-01-01'
end_date = '2025-07-10'

# get list of symbols
all_symbols_path = Path('Equities Universe - Symbols.csv')
all_symbols_data = pd.read_csv(all_symbols_path,index_col='symbol',parse_dates=['start_date', 'end_date'])
symbol = all_symbols_data.index.to_list()




# #check for duplicates in list:
# print(f'number of symbols: {len(symbol)}')
# print(f'number of unique symbols: {len(set(symbol))}')
# print(f'number of duplicates: {len(symbol) - len(set(symbol))}')

# # identify duplicates
# for i in range(len(symbol)):
#     for j in range(i + 1, len(symbol)):
#         if symbol[i] == symbol[j]:
#             print(f'Duplicate found at index {i} and {j}: {symbol[i]}')


data = get_symbol_data(symbol, start_date=START_DATE, end_date=end_date)
print(data)
#stats(data, start_date=START_DATE, end_date=end_date)
