import pandas as pd
from pathlib import Path
from get_data import get_symbol_data, symbol_data_to_returns_df
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import numpy as np
from charts import two_d_heatmap, three_d_plot




#pd.set_option('display.max_rows', None)

symbols = ['VOOG']
start_date = '2020-02-01'
end_date = '2020-02-10'

df = get_symbol_data(symbols, start_date, end_date)

portfolio_df = symbol_data_to_returns_df(df)

def portfolio_stats(
    df: pd.DataFrame = None,
):
    '''
    stats on portfolio_df
    they're meant to reflect stats of scoring logistic regression, 
    but instead of actual (1,0) vs predic (1,0) 
    it's comparing portfolio gains (gain, loss)


    
    '''
    print(f'portfolio_df: \n{portfolio_df.head()}')
    # n days close positive
    days_gains  = (portfolio_df > 0).sum().sum()
    print(f'gains: \n{days_gains}')
    # n days close negative
    days_losses = (portfolio_df <= 0).sum().sum()
    print(f'losses: \n{days_losses}')
    # pct days close positive
    pct_days_gains = days_gains / (days_gains + days_losses)
    print(f'pct_gains: \n{pct_days_gains}')
    # average gain on days there was a gain
    average_gain_per_gain_day = portfolio_df[portfolio_df > 0].mean().mean() / days_gains
    print(f'average_gain_per_gain_day: \n{average_gain_per_gain_day}')
    # pct average gain on days there was a gain
    pct_average_gain_per_gain_day = average_gain_per_gain_day / (days_gains + days_losses)
    print(f'pct_average_gain_per_gain_day: \n{pct_average_gain_per_gain_day}')


# two_d_heatmap(portfolio_df)




############### Plot
# Ensure the DataFrame is symmetrical
#


                
    