import pandas as pd
import numpy as np


def compare_returns(
        portfolio_df_1: pd.DataFrame = None, # Portfolio
        market_index: pd.DataFrame = None, # Index
        ):
        """
        input: (portfolio_df_1, portfolio_df_2)
        output: same_gain_loss_pct, same_gain_index_gain_pct, same_gain_portfolio_gain_pct, average_same_gain_pct
        """

        portfolio_df_1 = portfolio_df_1.applymap(np.sign)
        market_index = market_index.applymap(np.sign)

        # drop na
        portfolio_df_1 = portfolio_df_1.dropna(axis=0, how='all')
        market_index = market_index.dropna(axis=0, how='all')
        # N (counts)
        same_loss = ((portfolio_df_1 == -1) & (market_index == -1)).to_numpy().sum() #TN
        same_gain = ((portfolio_df_1 == 1) & (market_index == 1)).to_numpy().sum() #TP
        portfolio_gain = ((portfolio_df_1 == 1) & (market_index == -1)).to_numpy().sum() #FN
        index_gain = ((portfolio_df_1 == -1) & (market_index == 1)).to_numpy().sum() #FP

        #metrics
        same_gain_loss_pct  = (same_gain + same_loss) / (same_gain + same_loss + index_gain + portfolio_gain) if (same_gain + same_loss + index_gain + portfolio_gain) > 0 else 0 # accuracy
        same_gain_index_gain_pct  = same_gain / (same_gain + index_gain) if (same_gain + index_gain) > 0 else 0 # precision
        same_gain_portfolio_gain_pct  = same_gain / (same_gain + portfolio_gain) if (same_gain + portfolio_gain) > 0 else 0 # recall
        average_same_gain_pct  = 2 / ((1 / same_gain_index_gain_pct) + (1 / same_gain_portfolio_gain_pct)) if (same_gain_index_gain_pct + same_gain_portfolio_gain_pct) > 0 else 0 # f1
        portfolio_advantage = portfolio_gain / (portfolio_gain + same_gain) if (portfolio_gain + same_gain) > 0 else 0 # portfolio advantage
        portfolio_performance = portfolio_gain / index_gain if index_gain > 0 else 0 # portfolio performance
        portfolio_risk_of_loss = index_gain / (index_gain + same_loss) if (index_gain + same_loss) > 0 else 0 # portfolio risk of loss

        print(f'same_gain_loss_pct: {same_gain_loss_pct}')
        print(f'same_gain_index_gain_pct: {same_gain_index_gain_pct}')
        print(f'same_gain_portfolio_gain_pct: {same_gain_portfolio_gain_pct}')
        print(f'average_same_gain_pct: {average_same_gain_pct}')
        return same_gain_loss_pct, same_gain_index_gain_pct, same_gain_portfolio_gain_pct, average_same_gain_pct, portfolio_advantage

def stats(
        symbol_df: pd.DataFrame,
        first_series_symbol: str = None,
        start_date: str = None,
        end_date: str = None,
        ):
        """
        input: (symbol_df)
        output: quantity stats, quality stats
        """
        # initialize vars
        symbol_df = symbol_df.dropna(axis=0, how='all')

        # if first_series_symbol is provided, rename the first series
        if first_series_symbol:
               symbol_df.rename(columns={symbol_df.columns[0]: first_series_symbol}, inplace=True)
        else:
               pass
        
        if start_date is None or end_date is None:
            start_date = symbol_df.index[0]
            end_date = symbol_df.index[-1]
        else:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

        # set stats df index
        stats_index = symbol_df.columns.to_list()
        # Define dtypes dictionary (columns defined here)
        stats_dict = {
        'n_gain': 'Int64',
        'gain_ratio': 'float64',
        'relative_change': 'float64'
        }

        # Use the dictionary keys as column names
        stats_df = pd.DataFrame(
              index=stats_index,
              columns=list(stats_dict.keys())  # or dtypes_dict.keys()
              ).astype(stats_dict)

        # get subset of symbol_df for the given date range
        symbol_df = symbol_df.loc[start_date:end_date]

        # create change columns per symbol
        for col in stats_df.index:
            symbol_df[f'{col}_change'] = symbol_df[col].pct_change()
            symbol_df[f'{col}_change_b'] = np.where(symbol_df[f'{col}_change'] > 0, 1, -1)
            symbol_df[f'{col}_relative_change'] = symbol_df[col] / symbol_df[col].iloc[0]

            # calculate stats
            stats_df.loc[col, 'n_gain'] = symbol_df[f'{col}_change_b'].sum()
            stats_df.loc[col, 'gain_ratio'] = np.where(symbol_df[f'{col}_change_b'] == 1, 1, 0).sum() / len(symbol_df[f'{col}_change_b'])
            stats_df.loc[col, 'relative_change'] = round(symbol_df[f'{col}_relative_change'].iloc[-1],2) -1

        print(symbol_df.tail())
        print(stats_df)
        return stats_df