import pandas as pd
from pathlib import Path
from get_data import get_symbol_data, symbol_data_to_returns_df
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import numpy as np
from charts import timeline, returns_heatmap
from stats import stats 


#pd.set_option('display.max_rows', None)
START_DATE = '2020-02-01'
end_date = '2020-12-10'
symbol = ['TSLA', 'O', 'VOOG']

# Get data
#get_symbol_data(symbol, start_date=START_DATE, end_date=end_date)

# Read data
TSLA = pd.read_csv(Path(f'tsla.csv'), index_col='Date', parse_dates=True)
O = pd.read_csv(Path(f'O.csv'), index_col='Date', parse_dates=True)
VOOG = pd.read_csv(Path(f'VOOG.csv'), index_col='Date', parse_dates=True)

print(TSLA.loc[START_DATE:end_date, 'Close'].head())


# convert to returns df
TSLA_returns = symbol_data_to_returns_df(TSLA, start_date=START_DATE, end_date=end_date, value='change')
print(TSLA_returns.head())
TSLA_returns = symbol_data_to_returns_df(TSLA, start_date=START_DATE, end_date=end_date, value='change_b')

print(TSLA_returns.head())
#returns_heatmap(TSLA_returns)

# Create DataFrame for symbols
symbols_df = pd.DataFrame()
symbols_df['TSLA'] = pd.DataFrame(TSLA['Close'])
symbols_df['O'] = pd.DataFrame(O['Close'])
symbols_df['VOOG'] = pd.DataFrame(VOOG['Close'])

#timeline(symbol_data=symbols_df, start_date=START_DATE, end_date=end_date,y_axis='change')

#stats(symbols_df, start_date=START_DATE, end_date=end_date)



