from pathlib import Path
import pandas as pd
from datetime import date
from get_data import get_symbol_data
from charts import returns_heatmap, histogram
from get_data import symbol_data_to_returns_df
from stats import stats

#Parameters
portfolio_1 = ['TSLA'] #['TSLA','NVDA','META','AMZN','GOOG','MSFT','O','AAPL']
portfolio_2 = ['VOOG'] #['VOOG','VGT','VTI','SPY','QQQ','VOO']
start_date = '2020-01-01'
end_date = date.today().strftime('%Y-%m-%d')

# Assuming portfolio_1 and portfolio_2 are lists of symbols or data that get_symbol_data can process:
portfolios = [portfolio_1, portfolio_2]

# Dictionary to hold each processed DataFrame
processed_dfs = {}

for i, portfolio in enumerate(portfolios, start=1):
    # First, convert the list into a DataFrame using the get_symbol_data function
    df = get_symbol_data(portfolio, start_date=start_date, end_date=end_date)
    
    # Then apply your transformations using .pipe()
    df = (
        df
        .pipe(symbol_data_to_returns_df)
        .pipe(returns_heatmap)
        .pipe(histogram)
        .pipe(stats)
    )
    
    # Save the processed DataFrame in the dictionary. You can change the key as needed.
    processed_dfs[f'portfolio_{i}_df'] = df

# Now processed_dfs holds your transformed DataFrames, which you can use as needed:
portfolio_1_df = processed_dfs['portfolio_1_df']
portfolio_2_df = processed_dfs['portfolio_2_df']


