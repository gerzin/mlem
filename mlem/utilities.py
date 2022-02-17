import bz2
import pickle
from typing import Any, Dict, List, Tuple, Union, Iterable
import numpy as np
from numpy import ndarray
from pandas import concat
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV


def save_pickle(path: str, object: Any):
    """Saves a pickle file compressed in BZ2 format.

    Args:
        path (str): Path where to save object.
        object (Any): Object to save.
    """
    with bz2.open(path, "wb") as f:
        pickle.dump(object, f)


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
    "max_features": [5, "auto", "sqrt"],
    "min_samples_leaf": [10, 20, 50],
    "min_samples_split": [5, 10, 50],
    "n_estimators": [100, 350, 500],
    "criterion": ["gini", "entropy"],
}


def create_random_forest(
        x_train: ndarray,
        y_train: ndarray,
        hyperparameters: Dict[str, List[Any]] = __HYPERPARAMETERS,
) -> RandomForestClassifier:
    """Creates a random forest classifier via grid search.

    Args:
        x_train (ndarray): Training input examples.
        y_train (ndarray): Training target values.
        hyperparameters (Dict[str, List[Any]], optional): Dictionary of hyperparameters for the grid search. Defaults to the fixed ones.

    Returns:
        RandomForestClassifier: Random forest classifier.
    """
    rf = RandomForestClassifier()
    clf = RandomizedSearchCV(rf, hyperparameters, refit=True)
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
        DataFrame: Local attack dataset.
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


def get_frequencies(values: Union[Iterable, int, float]) -> np.array:
    """
    Given a set of values return an array containing the unique values and their respective frequencies.

    >>> a = np.array([1 1 1 2 2 2 3 4 4 5 4 5])
    >>> print(get_frequencies(a))
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
