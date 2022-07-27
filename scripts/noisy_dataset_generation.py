#!/usr/bin/env python3

from typing import List, Callable
import pandas as pd
import pickle
import numpy as np
from numpy import array
import argparse
from pathlib import Path


def insert_noise_numerical(dataset: pd.DataFrame, perc: float = 0.1,
                           noise_generating_function: Callable[[int], array] = np.random.normal):
    """
    Insert noise in a numerical dataset.

    Args:
        dataset (DataFrame): dataset on which to insert the noise.
        perc (float): percentage of noise in the range [0,1]
        noise_generating_function (Callable[[int], array]): function used to generate the noise, must take as input the number of noisy values to
                                   generate inside an argument named size and return an array containing the random values.


    Examples:

        >>> df = pd.DataFrame(data={'col1': 10 * [1], 'col2': 10 * [2], 'col3': 10 * [3]})
        >>> insert_noise(df, perc=0.1, noise_generating_function=np.random.rand) # note np.random.rand has a size parameter

    """
    n_rows, n_col = dataset.shape
    percentage = int(perc * n_rows)

    for c in range(n_col):
        index_to_replace = np.random.choice(dataset.index,
                                            size=percentage)
        new_values = noise_generating_function(size=percentage)
        assert (len(index_to_replace) == len(new_values))
        for ind, val in zip(index_to_replace, new_values):
            dataset.iloc[ind, c] = val


def frequency_based_noise(column, size):
    """
    Sample values from a column with replacement.

    Args:
        column: column to sample from
        size: number of elements to sample

    Returns:
        Array of samples
    """
    return column.sample(size, replace=True).to_numpy()


def insert_noise_categorical(dataset: pd.DataFrame, perc: float = 0.1,
                             noise_generating_function: Callable[[pd.Series, int], array] = frequency_based_noise):
    """

    Args:
        dataset (DataFrame): dataset on which to insert the noise ( it should only contain categorical variables )
        perc (float): percentage of noise in the range [0,1]
       noise_generating_function (Callable[[int], array]): function used to generate the noise, must take as input the number of noisy values to
                                   generate inside an argument named size and return an array containing the random values.

    Returns:

    """
    n_rows, n_col = dataset.shape
    percentage = int(perc * n_rows)

    for c in range(n_col):
        index_to_replace = np.random.choice(dataset.index,
                                            size=percentage)
        new_values = noise_generating_function(dataset[dataset.columns[c]], size=percentage)
        assert (len(index_to_replace) == len(new_values))
        for ind, val in zip(index_to_replace, new_values):
            dataset.iloc[ind, c] = val
