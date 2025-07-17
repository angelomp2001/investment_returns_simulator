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
        symbol_df: pd.DataFrame = None,
)
        """
        input: (symbol_df)
        output: quantity stats, quality stats
        """
        symbol = symbol_df.iloc[0]
        stats_df = pd.DataFrame(index=[symbol])
        symbol_df['change'] = symbol_df['close'].pct_change()
        symbol_df['b_change'] = 1 if symbol_df['change'] > 0 else 0
        symbol_df['relative_change'] = symbol_df['close'] / symbol_df['close'].iloc[0]
        stats_df[symbol, 'n_gain'] = symbol_df['b_change'].sum()
        stats_df[symbol, 'gain_ratio'] = symbol_df['change'].mean()