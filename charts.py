import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


def returns_heatmap(
        returns_df: pd.DataFrame = None,
    ):
    "2D heatmap of start to end dates and gains"

    # Create the heatmap
    plt.figure(figsize=(12, 8))

    # Create a custom colormap: red for negative, white for zero, green for positive.
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["red", "white", "green"], N=256)

    # Create a normalization instance that centers at zero.
    norm = TwoSlopeNorm(vmin=returns_df.min().min(), vcenter=0, vmax=returns_df.max().max())

    # Plot the heatmap
    ax = sns.heatmap(returns_df, annot=False, cmap=cmap, norm=norm, fmt=".2f", linewidths=0.5)

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
