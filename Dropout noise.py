import numpy as np
import pandas as pd


def add_dropout_noise(expression_matrix, dropout_rate):

    if isinstance(expression_matrix, pd.DataFrame):
        expression_matrix = expression_matrix.values

    dropout_mask = np.random.rand(*expression_matrix.shape) > dropout_rate

    noisy_expression_matrix = expression_matrix.copy()
    noisy_expression_matrix[~dropout_mask] = 0

    if isinstance(expression_matrix, pd.DataFrame):
        return pd.DataFrame(noisy_expression_matrix, index=expression_matrix.index, columns=expression_matrix.columns)
    else:
        return noisy_expression_matrix


dropout_rate = 0.6
example_expression_matrix = pd.read_csv("data\\Pollen_RSEMTopHat.csv", index_col=0)

noisy_expression_matrix = add_dropout_noise(example_expression_matrix, dropout_rate)
noisy_df = pd.DataFrame(noisy_expression_matrix)

noisy_df.to_csv('noisy_Pollen0.6.csv')

