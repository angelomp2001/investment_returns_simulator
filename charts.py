import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def two_d_heatmap(
        df: pd.DataFrame = None,
    ):
    "2D heatmap of start to end dates and gains"
    # Convert the index and columns to datetime for better formatting
    df.index = pd.to_datetime(df.index)
    df.columns = pd.to_datetime(df.columns)

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(df, annot=True, cmap='PiYG', fmt=".2f", linewidths=0.5)

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