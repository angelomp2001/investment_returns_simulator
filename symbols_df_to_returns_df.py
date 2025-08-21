# libraries
import pandas as pd
import numpy as np
import time as time

'helper'
def _standardize_input(input_object: pd.DataFrame | pd.Series) -> pd.DataFrame:
    '''
    Convert and clean pd.df/series to pd.df
    '''
    if isinstance(input_object, (pd.DataFrame, pd.Series)):
        pass
    else:
        raise ValueError("input_object must be a pandas DataFrame or Series.")

    # copy
    output_df = input_object.copy()

    # set to df
    if isinstance(output_df, pd.Series):
        output_df = pd.DataFrame(output_df, columns=[output_df.name])
    
    # set index to Datetime
    if not isinstance(output_df.index, pd.DatetimeIndex):
        output_df.index = pd.to_datetime(output_df.index)

    # sort
    output_df.sort_index()

    # dropna
    output_df = output_df.dropna(axis=0, how='all')

    return output_df
    
'helper'
def _calculate_values(df: pd.DataFrame, value: str) -> pd.DataFrame:
        '''
        calculate desired values using df. 
        df: provided input values
        value: desired output values: 'close', 'change', 'relative_change', 'change_sign', 'relative_change_sign'
        '''
        # vectorize df
        prices = df.to_numpy(dtype=float)
        
        # get input dimensions
        date_range, n_symbol = prices.shape

        # write mask of values (upper triangle only)
        contrib = np.triu(np.ones((date_range, date_range), dtype=bool), k=1)

        # Upper triangle index pairs
        i_idx, j_idx = np.triu_indices(date_range, k=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            if value == 'close':
                # end price only depends on j
                contrib = prices[j_idx, :]

            elif value == 'change':
                prev_prices = np.roll(prices, 1, axis=0)
                returns = (prices / prev_prices) - 1
                returns[0, :] = 0
                contrib = returns[j_idx, :]

            elif value is None or value == 'relative_change':
                contrib = (prices[j_idx, :] / prices[i_idx, :]) - 1

            elif value == 'change_sign':
                prev_prices = np.roll(prices, 1, axis=0)
                returns = (prices / prev_prices) - 1
                returns[0, :] = 0
                contrib = np.where(returns[j_idx, :] > 0, 1, -1)

            elif value == 'relative_change_sign':
                contrib = np.where((prices[j_idx, :] / prices[i_idx, :]) - 1 > 0, 1, -1)

            else:
                contrib = np.zeros((len(i_idx), n_symbol))

        # create output df
        # Initialize the upper triangle matrix
        matrix = np.zeros((date_range, date_range))

        # Sum across symbols (axis=1) into final upper-triangle matrix
        matrix[i_idx, j_idx] = contrib.sum(axis=1)

        # Fill lower triangle with NaN
        matrix[np.tril_indices(date_range, -1)] = np.nan

        return matrix

'helper'
def _standardize_output(np_array: np.array, index: pd.DatetimeIndex) -> pd.DataFrame:
    '''
    convert np.array to pd.df
    '''
    # convert matrix into df
    df = pd.DataFrame(np_array, index=index, columns=index)
    return df

'orchestrator'
def time_series_to_returns_df(
        time_series_1: pd.Series | pd.DataFrame,
        time_series_2: pd.Series = None,
        value: str
):
    '''
    converts time series symbols data to a returns df in four scenarios:
    1. 1 single series (pd.series/df) to returns df (pd.df)
    2. 1 multi series (pd.series/df) to returns df (pd.df)
    3. 2 single series (pd.series/df) to returns df (pd.df)
    4. 2 multi series (pd.series/df) to returns df (pd.df)
    '''
    # standardize input
    time_series_1 = _standardize_input(time_series_1)
    time_series_2 = _standardize_input(time_series_2) if time_series_2 is not None else None

    # calculate returns df values
    time_series_1_returns_array = _calculate_values(time_series_1, value = value)
    time_series_2_returns_array = _calculate_values(time_series_2, value = value) if time_series_2 is not None else None  
    
    # account for time series 2 is provided
    returns_array = time_series_1_returns_array - time_series_2_returns_array if time_series_2 is not None else time_series_1_returns_array

    # convert back to pd.df
    returns_df = _standardize_output(returns_array, index=time_series_1.index)

    #returns_df = returns_df.subtract(time_series_2_returns, fill_value=0) if time_series_2 is not None else returns_df

    return returns_df


'old function'
def symbols_df_to_returns_df(
    portfolio_1: pd.DataFrame = None,
    market_index: pd.DataFrame = None,
    start_date: str = None,
    end_date: str = None,
    value: str = None
):
    """
    Computes a returns matrix from one or two DataFrames.
    If both portfolio_1 and market_index are provided, returns the difference in returns.
    Each cell (i,j) is the return from date i to date j.
    portfolio_1: the main portfolio
    market_index: the market index portfolio_1 is compared against, if provided.
    start_date: start date of comparison
    end_date: end date of comparison
    value: what is returned: 'close', 'change', 'relative_change', 'change_sign', 'relative_change_sign'
    """
    
    def compute_returns(
            df: pd.DataFrame,
            start_date: str,
            end_date: str,
            value: str
        ):
        # dropna ✅
        df = df.dropna(axis=0, how='all')

        # set start and end dates
        if start_date is None:
            start_date = df.index.min()
        else:
            start_date = pd.to_datetime(start_date)

        if end_date is None:
            end_date = df.index.max()
        else:
            end_date = pd.to_datetime(end_date)

        # initialize vars
        df = df.loc[start_date:end_date]
        date_index = df.index
        m = len(date_index)
        returns_matrix = np.zeros((m, m)) # returns_matrix = np.full((m, m), np.nan)

        for symbol in df.columns:
            #vectorize prices
            prices = df[symbol].to_numpy(dtype=float)

            # Generate 2D grids: prices[i, j] = (price at j) / (price at i)
            start_prices = prices[:, np.newaxis]  # Shape (n,1)
            end_prices = prices[np.newaxis, :]    # Shape (1,n)

            with np.errstate(divide='ignore', invalid='ignore'):
                if value == 'close':
                    # contrib will be the close price on that day
                    contrib = np.tile(end_prices, (m, 1))
                elif value == 'change':
                    # new var which represents previous col (end_date) price
                    prev_end_date_prices = np.roll(prices, 1)
                    # calculate returns
                    returns = (prices / prev_end_date_prices) - 1
                    returns[0] = 0  # No previous day for first row
                    # tile the returns 
                    contrib = np.tile(returns, (m, 1))
                elif value is None or value == 'relative_change':
                    # this is the default
                    contrib = (end_prices / start_prices) - 1
                elif value == 'change_sign':
                    # new var which represents previous col (end_date) price
                    prev_end_date_prices = np.roll(prices, 1)
                    # calculate returns as binary (1= True, 0 <= False)
                    returns = ((prices / prev_end_date_prices) - 1) > 0
                    # first value does not exist
                    returns[0] = False
                    # replace True with 1, False with -1 
                    contrib = np.where(np.tile(returns, (m, 1)), 1, -1)
                elif value == 'relative_change_sign':
                    contrib = np.where((end_prices / start_prices) - 1 > 0, 1, -1)
                else:
                    contrib = np.zeros((m, m))

            # replace nan with 0
            # do not implement to not affect future calculations: contrib = np.nan_to_num(contrib)
            # Only upper triangle is valid (start < end)
            mask = np.triu(np.ones((m, m), dtype=bool), k=1)
            returns_matrix[mask] += contrib[mask]

        # NaN the lower triangle, not including main diagonal
        returns_matrix[np.tril_indices(m, -1)] = np.nan
        return pd.DataFrame(
            returns_matrix,
            index=date_index,
            columns=date_index
        )

    # Calculate returns for each DataFrame
    portfolio_1_returns = compute_returns(portfolio_1, start_date, end_date, value) if portfolio_1 is not None else None
    market_index_returns = compute_returns(market_index, start_date, end_date, value) if market_index is not None else None

    if portfolio_1_returns is not None and market_index_returns is not None:
        return portfolio_1_returns.subtract(market_index_returns, fill_value=0)

    if portfolio_1_returns is not None:
        return portfolio_1_returns
    elif market_index_returns is not None:
        return market_index_returns
    else:
        return None