import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from collections import Counter


def returns_heatmap(
        returns_df: pd.DataFrame = None
    ):
    '''
    input: returns_df.loc[start_date:end_date, symbol]
    output: 2D heatmap of start to end dates of values
    '''

    # Create a copy and format start/end to show only dates
    plot_df = returns_df.copy()
    plot_df.index = plot_df.index.strftime('%Y-%m-%d')
    plot_df.columns = plot_df.columns.strftime('%Y-%m-%d')

    # Create the heatmap
    plt.figure(figsize=(12, 8))

    # Create a custom colormap: red for negative, white for zero, green for positive.
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["red", "white", "green"], N=256)

    # Create a normalization instance that centers at zero.
    norm = TwoSlopeNorm(vmin=plot_df.min().min(), vcenter=0, vmax=plot_df.max().max())

    # Plot the heatmap
    ax = sns.heatmap(plot_df, annot=False, cmap=cmap, norm=norm, fmt=".2f", linewidths=0.5)

    # Add labels and title
    ax.set_xlabel('End Dates')
    ax.set_ylabel('Start Dates')
    ax.set_title('Portfolio Gains Heatmap')

    # Rotate x-tick labels for better readability
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()

def three_d_plot(
    df: pd.DataFrame = None,
):
    "3D plot of start to end dates and gains"
    # Initialize var
    df = df + df.T - np.diag(np.diag(df))
    
    # Convert indices to a format suitable for plotting
    x = np.array(pd.to_datetime(df.index), dtype='float')
    y = np.array(pd.to_datetime(df.columns), dtype='float')
    x, y = np.meshgrid(x, y)
    z = df.values

    # Create the 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a colormap from red to green
    cmap = plt.get_cmap('PiYG')

    # Plot the surface
    surface = ax.plot_surface(x, y, z, cmap=cmap, linewidth=0, antialiased=True, edgecolor='none')

    # Add color bar which maps values to colors
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

    # Set labels and title
    ax.set_xlabel('Start Dates')
    ax.set_ylabel('End Dates')
    ax.set_zlabel('Portfolio Gains')
    ax.set_title('3D Portfolio Gains Surface Plot')

    # Format the date labels on the axes
    ax.set_xticks(x[0])
    ax.set_xticklabels([item.strftime('%Y-%m-%d') for item in pd.to_datetime(df.index)], rotation=90)

    ax.set_yticks(y[0])
    ax.set_yticklabels([item.strftime('%Y-%m-%d') for item in pd.to_datetime(df.columns)], rotation=0)

    plt.tight_layout()
    plt.show()

def timeline(
    symbol_data: pd.DataFrame = None,
    start_date: str = None,
    end_date: str = None,
    symbol: str = None,
    y_axis: str = 'close',
):
    """
    Create a timeline change chart for a given symbol data.
    Shows how the price changes relative to the starting price.
    """
    # initialize vars
    start_date = pd.to_datetime(start_date) if start_date else None
    end_date = pd.to_datetime(end_date) if end_date else None
    start_date = start_date if start_date in symbol_data.index else symbol_data.index[0]
    end_date = end_date if end_date in symbol_data.index else symbol_data.index[-1]
    
    symbol_data = symbol_data.dropna(axis=0, how='all')
    symbol_data = symbol_data.loc[start_date:end_date]

    if symbol is None:
        column = symbol_data.columns[0]
    else:
        column = symbol

    # Slice data for given date range
    

    if y_axis == 'relative change':
        starting_price = symbol_data[column].iloc[0]
        symbol_data[column] = symbol_data[column] / starting_price
    else:
        y_axis == 'close'
    
    # Plot
    for col in symbol_data.columns:
        series = symbol_data[col].copy()

        if y_axis == 'change':
            starting_price = series.iloc[0]
            series = series / starting_price
            #plt.axhline(y=1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        plt.plot(series.index, series, marker='', linestyle='-', label=col)

    plt.title("Price Timeline" + (" (Relative to Start)" if y_axis == 'change' else ""))
    plt.xlabel("Date")
    plt.ylabel("Price Ratio" if y_axis == 'change' else "Price")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def histogram(
        data: pd.DataFrame or pd.Series = None,
        loc_start_date: str = None,
        loc_end_date: str = None,
        columns: list = None,
):
    '''
    Input: DataFrame or Series with float or categorical values.
    Output: Histogram: red (< 0), gray (= 0), green (> 0)
    '''
    if data is None:
        raise ValueError("Data cannot be None")

    ### incomplete code:
    # check if results_df
    is_upper_triangle = (
        data.shape[0] == data.shape[1]
        and (data.columns == data.index).all()
        and pd.api.types.is_datetime64_any_dtype(data.columns)
    )
    if not is_upper_triangle:
        pass
    ############## 
    
    # Handle DataFrame
    if isinstance(data, pd.DataFrame):
        if columns:
            data = data[columns]
        if loc_start_date and loc_end_date:
            data = data.loc[loc_start_date:loc_end_date]
        values = data.to_numpy().flatten()
    elif isinstance(data, pd.Series):
        if loc_start_date and loc_end_date:
            data = data.loc[loc_start_date:loc_end_date]
        values = data.to_numpy()
    else:
        raise TypeError("Input must be DataFrame or Series")

    # Remove NaNs
    values = values[~np.isnan(values)]

    # Check for discrete {-1, 0, 1} style data
    unique_vals = np.unique(values)
    is_discrete = np.all(np.isin(unique_vals, [-1, 0, 1]))

    plt.figure(figsize=(8, 6))

    if is_discrete:
        # Use bar chart for exact categories
        counts = Counter(values)
        heights = [counts.get(v, 0) for v in [-1, 0, 1]]
        colors = ['red', 'gray', 'green']
        plt.bar([-1, 0, 1], heights, color=colors, edgecolor='black')
        plt.xticks([-1, 0, 1])
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.title('Histogram of -1, 0, 1 values')
    else:
        # For continuous values, split into bins
        bins = np.histogram_bin_edges(values, bins='auto')
        plt.hist(values, bins=bins,
                 color='gray', edgecolor='black', alpha=0.5, label='All')

        # Overlay red bars for < 0
        neg_vals = values[values < 0]
        if len(neg_vals):
            plt.hist(neg_vals, bins=bins,
                     color='red', edgecolor='black', alpha=0.7, label='< 0')

        # Overlay green bars for > 0
        pos_vals = values[values > 0]
        if len(pos_vals):
            plt.hist(pos_vals, bins=bins,
                     color='green', edgecolor='black', alpha=0.7, label='> 0')

        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram (Red: < 0, Gray: 0, Green: > 0)')
        plt.legend()

    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
