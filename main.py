from pathlib import Path
import pandas as pd
from datetime import date
from get_data import get_symbol_data
from charts import two_d_heatmap
from stats import stats

#Parameters
portfolio_1 = ['TSLA'] #['TSLA','NVDA','META','AMZN','GOOG','MSFT','O','AAPL']
portfolio_2 = ['VOOG'] #['VOOG','VGT','VTI','SPY','QQQ','VOO']
start_date = '2020-01-01'
end_date = date.today().strftime('%Y-%m-%d')

df_1 = get_symbol_data(portfolio_1, start_date, end_date)
df_2 = get_symbol_data(portfolio_2, start_date, end_date)

stats(df_1, df_2)

#two_d_heatmap(df)