import bz2
import pickle
from typing import Any, Dict, List, Tuple, Union, Iterable
import numpy as np
from numpy import ndarray, concatenate
from pandas import concat
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.model_selection import RandomizedSearchCV
from numpy.core.fromnumeric import argmin
from numpy.core.shape_base import vstack
from numpy.lib.arraysetops import unique
from sklearn.utils import resample


def save_pickle_bz2(path: str, object: Any):
    """Saves a pickle file compressed in BZ2 format.

    Args:
        path (str): Path where to save object.
        object (Any): Object to save.
    """
    with bz2.open(path, "wb") as f:
        pickle.dump(object, f)


def load_pickle_bz2(path):
    """
    Loads data saved with save_pickle_bz2

    Args:
        path (str): Path where the data are located

    Returns:
        loaded object
    """
    with bz2.BZ2File(path) as f:
        data = pickle.load(f)
    return data


def save_txt(path: str, txt: str):
    """Saves a text file on disk.

    Args:
        path (str): Path where to save the string.
        txt (str): String to save.
    """
    with open(path, "w") as f:
        f.write(txt)


def save_train_test_datasets(path: str, x_train: ndarray, y_train: ndarray, x_test: ndarray, y_test: ndarray) -> None:
    """
    Saves on a compressed file the training and test sets with the respective labels.
    Args:
        path:
        x_train: training features.
        y_train: training targets.
        x_test: test features.
        y_test: test targets.

    Returns:
        None
    """
    np.savez_compressed(
        path,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )


def load_train_test_datasets(path: str) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Loads training and test sets, with the respective labels, saved with save_train_test_datasets.
    Args:
        path: path of the file containing the datasets.

    Returns:
        (x_train: ndarray, y_train: ndarray, x_test: ndarray, y_test: ndarray) - training features, training targets, test features, test targets
    """
    loaded = np.load(path, allow_pickle=True)
    x_train: ndarray = loaded["x_train"]
    y_train: ndarray = loaded["y_train"]
    x_test: ndarray = loaded["x_test"]
    y_test: ndarray = loaded["y_test"]

    return x_train, y_train, x_test, y_test


# Hyperparameters of the grid search
__HYPERPARAMETERS = {
    "bootstrap": [True, False],
    "max_depth": [100, 350, 500],
    "max_features": ["auto", "sqrt"],
    "min_samples_leaf": [10, 20, 50],
    "min_samples_split": [5, 10, 50],
    "n_estimators": [100, 350, 500],
    "criterion": ["gini", "entropy"],
}


def create_random_forest(
        x_train: ndarray,
        y_train: ndarray,
        hyperparameters: Dict[str, List[Any]] = __HYPERPARAMETERS,
        n_jobs=4
) -> RandomForestClassifier:
    """Creates a random forest classifier via grid search.

    Args:
        x_train (ndarray): Training input examples.
        y_train (ndarray): Training target values.
        hyperparameters (Dict[str, List[Any]], optional): Dictionary of hyperparameters for the grid search. Defaults to the fixed ones.
        n_jobs: Number of jobs to run in parallel in the grid search. (default 4)

    Returns:
        RandomForestClassifier: Random forest classifier.
    """

    rf = RandomForestClassifier()
    clf = RandomizedSearchCV(rf, hyperparameters, refit=True, n_jobs=n_jobs, verbose=0)
    clf.fit(x_train, y_train)
    return clf.best_estimator_


def __create_local_attack_dataset(
        y_prob: ndarray,
        y: ndarray,
        inout: str,
) -> DataFrame:
    """Creates a dataframe of the form (y_prob, y, "in" / "out") to be used for
    the creation of a dataset for the Attack Model of the Membership Inference.



    Args:
        y_prob (ndarray): Prediction probability for the shadow model.
        y (ndarray): Target label predicted by the shadow model.
        inout (str): Whether these data come from the training test ("in") or test set ("out").

    Returns:
        DataFrame: Local attack dataset. The dataframe contains one column for each class with its respective probability,
        plus a column "label" with the real label and a column "inout" with the value "in" or "out"
    """
    df = DataFrame(y_prob)
    df["label"] = y
    df["inout"] = inout
    return df


def create_attack_dataset(
        y_prob_train: ndarray,
        y_train: ndarray,
        y_prob_test: ndarray = None,
        y_test: ndarray = None,
) -> DataFrame:
    """Creates a dataset for the Membership Inference Attack Model.
    Creates an attack dataset by creating tuples of the form (y_prob_train, y_train, "in") for the
    data belonging to the train set of the model we are trying to attack and (y_prob_test, y_test, "out")
    for the data in the test set of that model.
    Args:
        y_prob_train (ndarray): Predicted probabilities on the training set.
        y_train (ndarray): Labels of the training set.
        y_prob_test (ndarray, optional): Predicted probabilities on the test set.
        y_test (ndarray, optional): Labels of the test set.
    Returns:
        DataFrame: Local attack model.
    """
    df_in: DataFrame = __create_local_attack_dataset(y_prob_train, y_train, "in")
    if y_prob_test is not None and y_test is not None:
        df_out: DataFrame = __create_local_attack_dataset(y_prob_test, y_test, "out")
        return concat([df_in, df_out])
    return df_in


def minority_class_resample(x: ndarray, y: ndarray, n_samples: int = 50) -> Tuple[ndarray, ndarray]:
    """Resamples x and y generating n_samples elements of the minority class.

    Args:
        x (ndarray): Input data.
        y (ndarray): Target labels.
        n_samples (int, optional): Number of samples to be drawn for the minority class. Defaults to 50.

    Returns:
        Tuple[ndarray, ndarray]: Resampled x and y.
    """
    # Unique classes and their frequencies
    classes, occurrences = unique(y, return_counts=True)
    # Minority class
    min_class: int = classes[argmin(occurrences)]
    # Records of every other class
    x_maj, y_maj = x[y != min_class], y[y != min_class]
    # Records of the minority class
    x_min, y_min = x[y == min_class], y[y == min_class]
    # Resamples the minority class
    x_min, y_min = resample(x_min, y_min, n_samples=n_samples)
    # Reconstructs the original dataset
    x = vstack((x_maj, x_min))
    y = concatenate((y_maj, y_min))
    return x, y


def frequencies(values: Union[Iterable, int, float]) -> np.array:
    """
    Given a set of values return an array containing the unique values and their respective frequencies.

    >>> a = np.array([1 1 1 2 2 2 3 4 4 5 4 5])
    >>> print(frequencies(a))
    >>> array([[1, 3],
    >>>       [2, 3],
    >>>       [3, 1],
    >>>       [4, 3],
    >>>       [5, 2]])

    Args:
        values: array or list from which to extract the unique values and their frequencies
    Returns:
        array - array of [unique value, its frequency] pairs
    """
    # unique, counts
    u, c = np.unique(values, return_counts=True)
    return np.asarray((u, c)).T
