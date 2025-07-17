import pandas as pd
from pathlib import Path
from get_data import get_symbol_data, symbol_data_to_returns_df
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import numpy as np
from charts import timeline
from stats import compare_returns


#pd.set_option('display.max_rows', None)
START_DATE = '2020-02-01'
end_date = '2020-12-10'
symbol = 'TSLA'

TSLA = pd.read_csv(Path(f'tsla.csv'), index_col='Date', parse_dates=True)
O = pd.read_csv(Path(f'O.csv'), index_col='Date', parse_dates=True)

symbol_df = pd.DataFrame()
symbol_df['TSLA'] = pd.DataFrame(TSLA['Close'])
symbol_df['O'] = pd.DataFrame(O['Close'])

timeline(symbol_data=symbol_df, start_date=START_DATE, end_date=end_date,y_axis='change')



