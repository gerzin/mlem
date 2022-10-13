"""
Utilities to load the adult blackbox and data to use for the experiments.

The data are in experiments/datasets/adult
"""
from pathlib import Path
import numpy as np
import sys

sys.path.append("../..")
from mlem.utilities import load_pickle_bz2

DEFAULT_DIVA_PATH = Path(__file__).parent.parent.parent / "datasets" / "diva"


def load_diva_data(data_name="diva-blackbox-data.npz", data_folder_path=None):
    """
    Load the diva data dictionary in experiments/data/adult
    Args:
        data_name: name of the numpy dict. containing the data
        data_folder_path: path of the folder containing the data

    Returns:
        npz object containing all the data
    """
    if data_folder_path:
        adult_path = data_folder_path
    else:
        adult_path = DEFAULT_DIVA_PATH

    return np.load(adult_path / data_name)


def load_diva_randomforest(data_name="diva_randfor.bz2", data_folder_path=None):
    """
    Load the random forest trained on adult
    Args:
        data_name: name of the random forest file.
        data_folder_path: path of the folder containing the data

    Returns:
        RandomForest trained on the adult data.
    """
    if data_folder_path:
        adult_path = data_folder_path
    else:
        adult_path = DEFAULT_DIVA_PATH

    return load_pickle_bz2(adult_path / data_name)