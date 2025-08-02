import pandas as pd
import numpy as np


def compare_returns(
        portfolio_df_1: pd.DataFrame = None, # Portfolio
        market_index: pd.DataFrame = None, # Index
        ):
        """
        input: 1 or 2: symbols_df.loc[start_date:end_date, symbol]
        output: same_gain_loss_pct, same_gain_index_gain_pct, same_gain_portfolio_gain_pct, average_same_gain_pct
        """
        # convert magnitude to sign (negative = -1, 0=0, positive = 1)
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

def symbols_stats(
        symbol_df: pd.DataFrame, #symbol_data
        first_series_symbol: str = None,
        start_date: str = None,
        end_date: str = None,
        ):
    """
    input: (symbol_df: DataFrame or Series)
    output: quantity stats, quality stats
    ### there is an issue when running this function immediately after downloading NEW data.  Fix: run it again. ###
    """
    d = 60 # number of consecutive days required to define consistent gain
    # If symbol_df is a Series, convert it into a DataFrame
    if isinstance(symbol_df, pd.Series):
        symbol_df = symbol_df.to_frame()
    
    # Ensure that the index is datetime if possible
    if not pd.api.types.is_datetime64_any_dtype(symbol_df.index):
        try:
            symbol_df.index = pd.to_datetime(symbol_df.index)
        except Exception as e:
            raise ValueError("Index must be datetime-like or convertible to datetime.") from e

    # Drop rows with all NaN values
    symbol_df = symbol_df.dropna(axis=0, how='all')

    # If first_series_symbol is provided, rename the first series
    if first_series_symbol:
        symbol_df.rename(columns={symbol_df.columns[0]: first_series_symbol}, inplace=True)
    
    # Set date range bounds if not explicitly provided
    if start_date is None:
        start_date = symbol_df.index[0]
    else:
        start_date = pd.to_datetime(start_date)

        if end_date is None:
            end_date = symbol_df.index[-1]
        else:
            end_date = pd.to_datetime(end_date)

    # Subset the symbol_df for the given date range
    symbol_df = symbol_df.loc[start_date:end_date]

    # set stats df index (each symbol is one column)
    stats_index = symbol_df.columns.to_list()

    # Define dtypes dictionary (columns defined here)
    stats_dict = {
        'n_change': 'Int64',
        'n_gain': 'Int64',
        'gain_ratio': 'float64',
        'd': 'Int64',
        'd_gain_b': 'Int64',
        'd_gain_b_ratio': 'float64',
        'relative_change': 'float64'
    }

    # Create a DataFrame to hold stats for each symbol
    stats_df = pd.DataFrame(
        index=stats_index,
        columns=list(stats_dict.keys())
    ).astype(stats_dict)

    # Create change columns in symbol_df per symbol and calculate the statistics
    for col in stats_index:
        symbol_df[f'{col}_change'] = symbol_df[col].pct_change(fill_method=None)
        symbol_df[f'{col}_change_b'] = np.where(symbol_df[f'{col}_change'] > 0, 1, -1)
        symbol_df[f'{col}_relative_change'] = symbol_df[col] / symbol_df[col].iloc[0]

        # Calculate stats
        stats_df.loc[col, 'n_change'] = len(symbol_df[f'{col}_change'])
        stats_df.loc[col, 'n_gain'] = (symbol_df[f'{col}_change_b'] > 0).sum()
        stats_df.loc[col, 'gain_ratio'] = round((symbol_df[f'{col}_change_b'] == 1).sum() / len(symbol_df[f'{col}_change_b']),2)
        stats_df.loc[col, 'd'] = d
        stats_df.loc[col, 'd_gain_b'] = (symbol_df[-d:][f'{col}_change_b'] == 1).sum()
        stats_df.loc[col, 'd_gain_b_ratio'] = round((symbol_df[-d:][f'{col}_change_b'] == 1).sum() / d,2)
        stats_df.loc[col, 'relative_change'] = round(symbol_df[f'{col}_relative_change'].iloc[-1],2) - 1

    print(stats_df)
    return stats_df

def symbols_and_results_stats(
    symbol_df: pd.DataFrame,
    label: str = None,
    start_date: str = None,
    end_date: str = None,
    t: int = None
):
    """
    Computes stats for time series price data or upper triangle matrix of pairwise changes.

    Parameters:
        symbol_df: DataFrame of prices (columns = symbols, index = dates)
                   or square upper triangle matrix (index = columns = dates)
        first_series_symbol: Optional label rename for the first series
        start_date, end_date: Optional date bounds

    Returns:
        stats_df: DataFrame with summary statistics
    """
    print(f't: {t}')

    t = t  # trailing window for d-day stats

    # Handle Series input
    if isinstance(symbol_df, pd.Series):
        symbol_df = symbol_df.to_frame()

    # Ensure datetime index
    if not pd.api.types.is_datetime64_any_dtype(symbol_df.index):
        try:
            symbol_df.index = pd.to_datetime(symbol_df.index)
        except Exception as e:
            raise ValueError("Index must be datetime-like or convertible to datetime.") from e

    # Drop all-NaN rows
    symbol_df = symbol_df.dropna(axis=0, how='all')

    # Check if this is an upper triangle matrix (square DataFrame with datetime index and columns)
    is_upper_triangle = (
        symbol_df.shape[0] == symbol_df.shape[1]
        and (symbol_df.columns == symbol_df.index).all()
        and pd.api.types.is_datetime64_any_dtype(symbol_df.columns)
    )

    # If this is an upper triangle matrix of changes or scores
    if is_upper_triangle:
        if not label:
            label = "results_df"

        stats_dict = {}

        # Full upper triangle
        # create upper triangle mask
        upper_mask = np.triu(np.ones(symbol_df.shape, dtype=bool), k=1) # diagonal not included b/c it's NaN for change values
        
        # save upper values of symbol_df
        upper_values = symbol_df.values[upper_mask]
        
        # remove NaN
        upper_values_nonan = upper_values[~np.isnan(upper_values)]

        # stats: number of values
        stats_dict['n'] = len(upper_values_nonan)

        # stats: number of change-increase
        stats_dict['n_gain'] = (upper_values_nonan > 0).sum()

        # stats: number of change-decrease
        stats_dict['n_loss'] = (upper_values_nonan <= 0).sum()

        #stats: ratio of change-increase
        stats_dict['gain_ratio'] = round(stats_dict['n_gain'] / stats_dict['n'], 2) if stats_dict['n'] > 0 else np.nan

        # Trailing d-day upper triangle block
        if t <= symbol_df.shape[0]:
            # isolate only start-end days
            trailing_block = symbol_df.values[-t:, -t:]

            # create upper triangle mask
            trailing_mask = np.triu(np.ones((t, t), dtype=bool), k=1)

            # extract upper triangle mask values from symbol_df
            trailing_values = trailing_block[trailing_mask]

            # remove NaN
            trailing_values_nonan = trailing_values[~np.isnan(trailing_values)]

            # stats: days scope
            stats_dict['t'] = t

            # stats: number of change-increase in days
            stats_dict['t_gain_sign'] = (trailing_values_nonan > 0).sum()

            #stats: ratio of change-increase in days
            stats_dict['t_gain_sign_ratio'] = round(stats_dict['t_gain_sign'] / len(trailing_values_nonan), 2) if len(trailing_values_nonan) > 0 else np.nan
        else:
            stats_dict['t'] = t
            stats_dict['t_gain_sign'] = np.nan
            stats_dict['t_gain_sign_ratio'] = np.nan

        # stats: max number of days between gains (row-wise, in upper triangle)
        # create a matrix where the gains will be
        gain_matrix = (symbol_df.values > 0)    # True where gain, otherwise False
        
        # get number of rows
        m = symbol_df.shape[0]

        # initialize var that will log the max gap between days of gains
        max_gap = 0

        # initialize var that will log the start and end date of max_gap
        max_gap_coords = None   # (row, start_col, end_col)

        # loop through each row to collect max days btwn gains
        for row in range(m):
            # save indices of gains relative to slice
            slice_indices = np.where(gain_matrix[row, row+1:])[0]

            # adjust slice index to match gains matrix index
            gain_indices = slice_indices + (row+1)

            # ensure there are 2+ gains in order to calculate a distance
            if len(gain_indices) >= 2:
                # distance between gain indices.  0 days = consecutive days. 
                gaps = np.diff(gain_indices) - 1
                # print(f'gaps: {gaps}') = [0 0 0 0 0 ...]
                # save biggest gap between days
                row_max_gap = gaps.max()

                # save row max gap as max gap if it's bigger than max gap 
                # print(f'row_max_gap: {row_max_gap}') = 0
                if row_max_gap > max_gap:
                    max_gap = row_max_gap
                    
                    # Get the index of max gap
                    max_gap_idx = gaps.argmax()

                    # lookup max gap index in gain indices to get/save start col of the max gap
                    start_col = gain_indices[max_gap_idx]

                    # look up the index after the max gap index in gain indices and save it as the end col of first max gap 
                    end_col = gain_indices[max_gap_idx + 1]

                    # save row, start col, and end col, as location of the max gap
                    max_gap_coords = (row, start_col, end_col)
                    
        # the largest number of days between gain days
        stats_dict['max_days_btwn_gains'] = int(max_gap)

        # the corresponding start and end date of the largest number of days between gain days
        if max_gap_coords:
            _, max_start_date, max_end_date = max_gap_coords
            stats_dict['max_days_btwn_gains_coords'] = (\
            symbol_df.columns[max_start_date].strftime('%Y-%m-%d'),
            symbol_df.columns[max_end_date].strftime('%Y-%m-%d')
            )
        else:
            stats_dict['max_days_btwn_gains_coords'] = (None, None)

        
        # stats: max number of consecutive days of gains (row-wise in upper triangle)

        max_streak = 0                  # overall max streak length
        current_streak = 0              # streak counter for current row
        max_streak_coords = None        # will hold (row, start_col, end_col) of max streak

        for row in range(m):
            streak_start = None
            current_streak = 0

            # iterate across upper triangle (cols after current row)
            for col in range(row + 1, m):
                if gain_matrix[row, col]:  # True if there's a gain
                    if current_streak == 0:
                        streak_start = col  # mark beginning of a streak
                    current_streak += 1

                    # update max streak if needed
                    if current_streak > max_streak:
                        max_streak = current_streak
                        max_streak_coords = (row, streak_start, col)
                else:
                    current_streak = 0  # reset streak if no gain

        # translate max streak indices to dates
        if max_streak_coords:
            _, start_col, end_col = max_streak_coords
            max_streak_start_date = symbol_df.columns[start_col].strftime('%Y-%m-%d')
            max_streak_end_date = symbol_df.columns[end_col].strftime('%Y-%m-%d')
        else:
            max_streak_start_date = None
            max_streak_end_date = None

        # largest number of consecutive days of gains
        stats_dict['max_consecutive_days_of_gains'] = int(max_streak)

        # the corresponding start and end date of the largest number of consecutive days of gains
        stats_dict['max_consecutive_days_of_gains_coords'] = (max_streak_start_date, max_streak_end_date)

        # return stats as df.
        return pd.DataFrame([stats_dict], index=[label])

    # === Handle time series format ===

    # Rename first column if requested
    if label:
        symbol_df.rename(columns={symbol_df.columns[0]: label}, inplace=True)

    # Date bounds
    if start_date is None:
        start_date = symbol_df.index[0]
    else:
        start_date = pd.to_datetime(start_date)

    if end_date is None:
        end_date = symbol_df.index[-1]
    else:
        end_date = pd.to_datetime(end_date)

    symbol_df = symbol_df.loc[start_date:end_date]

    stats_index = symbol_df.columns.to_list()
    stats_dtypes = {
        'n_change': 'Int64',
        'n_gain': 'Int64',
        'gain_ratio': 'float64',
        'relative_change': 'float64',
        't': 'Int64',
        't_gain_sign': 'Int64',
        't_gain_sign_ratio': 'float64',
        't_relative_change': 'float64',
        't_harmonic_mean': 'float64',
        't_rank_harmonic_mean': 'Int64'
    }

    stats_df = pd.DataFrame(index=stats_index, columns=list(stats_dtypes.keys())).astype(stats_dtypes)

    ## test 1/2
    # Prepare new columns in a dictionary
    change_cols = {}
    change_sign_cols = {}
    relative_change_cols = {}

    for col in stats_index:
        pct_change = symbol_df[col].pct_change(fill_method=None)
        change_sign = np.where(pct_change > 0, 1, -1)
        rel_change = symbol_df[col] / symbol_df[col].iloc[0]

        change_cols[f'{col}_change'] = pct_change
        change_sign_cols[f'{col}_change_sign'] = change_sign
        relative_change_cols[f'{col}_relative_change'] = rel_change

        # test 2/2
    # Concatenate all at once
    new_cols_df = pd.DataFrame({**change_cols, **change_sign_cols, **relative_change_cols}, index=symbol_df.index)
    # Join with symbol_df only once (copy recommended for defragmenting)
    symbol_df = pd.concat([symbol_df, new_cols_df], axis=1).copy()
    print(symbol_df['CRCL'].describe())


    for col in stats_index:
        # symbol_df[f'{col}_change'] = symbol_df[col].pct_change()
        # symbol_df[f'{col}_change_sign'] = np.where(symbol_df[f'{col}_change'] > 0, 1, -1)
        # symbol_df[f'{col}_relative_change'] = symbol_df[col] / symbol_df[col].iloc[0]

        stats_df.loc[col, 'n_change'] = symbol_df[f'{col}_change'].count()
        stats_df.loc[col, 'n_gain'] = (symbol_df[f'{col}_change_sign'] == 1).sum()

        stats_df.loc[col, 'gain_ratio'] = (
        round(stats_df.loc[col, 'n_gain'] / stats_df.loc[col, 'n_change'], 2)
        if stats_df.loc[col, 'n_change'] > 0 else np.nan
    )
        stats_df.loc[col, 'relative_change'] = round(symbol_df[f'{col}_relative_change'].iloc[-1], 2) - 1
        stats_df.loc[col, 't'] = t
        stats_df.loc[col, 't_gain_sign'] = (symbol_df[f'{col}_change_sign'].iloc[-t:] == 1).sum()
        stats_df.loc[col, 't_gain_sign_ratio'] = round(stats_df.loc[col, 't_gain_sign'] / t, 2)

        # assuming input is symbol_df
        print(f"{col}-1: {symbol_df[f'{col}_relative_change'].iloc[-1]}")
        print(f"{col}_-t: {symbol_df[f'{col}_relative_change'].iloc[-t]}")
        stats_df.loc[col, 't_relative_change'] = round(symbol_df[f'{col}_relative_change'].iloc[-1], 2) / round(symbol_df[f'{col}_relative_change'].iloc[-t],2) - 1
                
        # === harmonic mean of trailing change sign ratio ===
        # === harmonic mean of t_gain_sign and t_relative_change ===
        t_gain_sign = stats_df.loc[col, 't_gain_sign']
        t_relative_change = stats_df.loc[col, 't_relative_change']
        if t_gain_sign > 0 and t_relative_change > 0:
            harmonic_mean = 2 / ((1 / t_gain_sign) + (1 / t_relative_change))
            stats_df.loc[col, 't_harmonic_mean'] = round(harmonic_mean, 4)
        else:
            stats_df.loc[col, 't_harmonic_mean'] = 0.0



    # Rank harmonic_mean (1 = best)
    stats_df['t_rank_harmonic_mean'] = stats_df['t_harmonic_mean'].rank(method='min', ascending=False).astype('Int64')

    #sort
    stats_df = stats_df.sort_values('t_rank_harmonic_mean')

    #print
    print(stats_df.head(30))
    print(list(stats_df.head(50).index))

    return stats_df
