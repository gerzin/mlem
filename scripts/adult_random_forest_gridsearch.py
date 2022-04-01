#!/usr/bin/env python3
import bz2
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def save_pickle_bz2(path: str, obj):
    """Saves a pickle file compressed in BZ2 format.

    Args:
        path (str): Path where to save object.
        obj (Any): Object to save.
    """
    with bz2.open(path, "wb") as f:
        pickle.dump(obj, f)


ROOT = Path(__file__).parent.parent / "data" / "adult"
TRAINSET_PATH = ROOT / "test" / "test.csv"
TESTSET_PATH = ROOT / "train" / "train.csv"

print(f"{TRAINSET_PATH=}")
print(f"{TRAINSET_PATH=}")

test = pd.read_csv(TESTSET_PATH).to_numpy()
train = pd.read_csv(TRAINSET_PATH).to_numpy()

X_train, y_train = train[:, :-1], train[:, -1].astype(int)
X_test, y_test = test[:, :-1], test[:, -1].astype(int)

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
        x,
        y,
        hyperparameters=__HYPERPARAMETERS,
        n_jobs=4
) -> RandomForestClassifier:
    """Creates a random forest classifier via grid search.

    Args:
        x (ndarray): Training input examples.
        y (ndarray): Training target values.
        hyperparameters (Dict[str, List[Any]], optional): Dictionary of hyperparameters for the grid search. Defaults to the fixed ones.
        n_jobs: Number of jobs to run in parallel in the grid search. (default 4)

    Returns:
        RandomForestClassifier: Random forest classifier.
    """

    rf = RandomForestClassifier()
    clf = HalvingGridSearchCV(rf, hyperparameters, refit=True, n_jobs=n_jobs, verbose=1)
    clf.fit(x, y)
    return clf.best_estimator_


if __name__ == '__main__':
    rand_for = create_random_forest(X_train, y_train)
    preds = rand_for.predict(X_test)
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))

    np.savez_compressed(
        "./adult_randfor.data",
        x_train=X_train,
        x_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    save_pickle_bz2("./adult_randfor.bz2", rand_for)
