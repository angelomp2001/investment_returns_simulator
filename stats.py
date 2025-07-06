import pandas as pd
import numpy as np


def stats(
        portfolio_df_1: pd.DataFrame = None, # Portfolio
        portfolio_df_2: pd.DataFrame = None, # Index
        ):
        """
        input: (portfolio_df_1, portfolio_df_2)
        output: accuracy, precision, recall, f1_score
        """

        portfolio_df_1 = portfolio_df_1.applymap(np.sign)
        portfolio_df_2 = portfolio_df_2.applymap(np.sign)

        # N (counts)
        TN = ((portfolio_df_1 == -1) & (portfolio_df_2 == -1)).to_numpy().sum()
        TP = ((portfolio_df_1 == 1) & (portfolio_df_2 == 1)).to_numpy().sum()
        FN = ((portfolio_df_1 == 1) & (portfolio_df_2 == -1)).to_numpy().sum()
        FP = ((portfolio_df_1 == -1) & (portfolio_df_2 == 1)).to_numpy().sum()

        #metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 / ((1 / precision) + (1 / recall)) if (precision + recall) > 0 else 0
        
        print(f'accuracy: {accuracy}')
        print(f'precision: {precision}')
        print(f'recall: {recall}')
        print(f'f1: {f1}')
        return accuracy, precision, recall, f1