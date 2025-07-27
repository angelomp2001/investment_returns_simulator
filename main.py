from pathlib import Path
import pandas as pd
from datetime import date
from get_data import get_symbol_data
from charts import returns_heatmap, histogram
from get_data import symbol_data_to_returns_df
from stats import stats

#Parameters
portfolio_1 = ['TSLA'] #['TSLA','NVDA','META','AMZN','GOOG','MSFT','O','AAPL']
market_index = ['VOOG'] #['VOOG','VGT','VTI','SPY','QQQ','VOO']
start_date = '2020-01-01'
end_date = date.today().strftime('%Y-%m-%d')

# Assuming portfolio_1 and market_index are lists of symbols or data that get_symbol_data can process:
portfolios = [portfolio_1, market_index]


# First, convert the list into a DataFrame using the get_symbol_data function
portfolio = (
    portfolio
    .pipe(get_symbol_data, start_date=start_date, end_date=end_date)
    .pipe(symbol_data_to_returns_df, market_index=market_index, start_date=start_date, end_date=end_date)
    .pipe(stats)
)

    # Save the processed DataFrame in the dictionary. You can change the key as needed.
    processed_dfs[f'portfolio_{i}_df'] = df

# Now processed_dfs holds your transformed DataFrames, which you can use as needed:
portfolio_1_df = processed_dfs['portfolio_1_df']
portfolio_2_df = processed_dfs['portfolio_2_df']


