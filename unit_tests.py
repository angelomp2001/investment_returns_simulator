import pandas as pd
from pathlib import Path
from get_data import get_symbol_data, symbol_data_to_returns_df
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import numpy as np
from charts import two_d_heatmap, three_d_plot
from stats import stats


#pd.set_option('display.max_rows', None)
start_date = '2020-02-01'
end_date = '2020-12-10'

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

two_d_heatmap(net_returns)

print(f'\n')
