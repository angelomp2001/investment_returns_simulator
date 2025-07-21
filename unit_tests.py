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
symbol = 'AAPL'

## test: only save symbols_data_xxxx when symbol > 1.
## test: lookup symbols_data before processing get_symbol_data

data = get_symbol_data(symbol, start_date=START_DATE, end_date=end_date)

