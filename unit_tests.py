import pandas as pd
from pathlib import Path
from get_data import get_symbol_data, symbol_data_to_returns_df
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import numpy as np
from charts import two_d_heatmap, three_d_plot
from stats import stats


#pd.set_option('display.max_rows', None)
start_date = '2020-02-01'
end_date = '2025-02-10'

symbols = ['MSFT']
df = get_symbol_data(symbols, start_date, end_date)
portfolio_df_1 = symbol_data_to_returns_df(df)


symbols = ['VOOG']
df = get_symbol_data(symbols, start_date, end_date)
portfolio_df_2 = symbol_data_to_returns_df(df)

stats(portfolio_df_1, portfolio_df_2)

print(f'\n')
