#!/usr/bin/env python3
import argparse
from typing import Sequence, Any, Tuple

import numpy as np
import sys

sys.path.append("../lime")
sys.path.append("../")
from pathlib import Path
from numpy import ndarray

from mlem.black_box import BlackBox, SklearnBlackBox
from mlem.ensemble import EnsembleClassifier
from mlem.enumerators import SamplingTechnique
from mlem.utilities import load_pickle_bz2

from lime.lime_tabular import LimeTabularExplainer
from joblib import Parallel, delayed, cpu_count


def __get_local_data(
        x: ndarray,
        y: ndarray,
        exp: LimeTabularExplainer,
        black_box: BlackBox,
        sampling_method: SamplingTechnique,
        num_samples: int,
        labels: Sequence[Any],
) -> Tuple[EnsembleClassifier, ndarray, ndarray]:
    """
        Args:
            x:
            y:
            exp:
            black_box:
            sampling_method:
            num_samples:
            labels:
        Returns:
            the local model, the neighborhood and the predictions on the neighborhood made by the local model.
        """
    # Exploits Lime to get the neighborhood and the local model
    _, models, x_neigh = exp.explain_instance(
        x,
        black_box.predict_proba,
        labels=labels,
        sampling_method=sampling_method,
        num_samples=num_samples,
        num_features=len(x),
    )
    # Local model is the one pointed by the instance
    local_model = EnsembleClassifier(regressors=models)
    # Generates predictions for the neighborhood
    y_neigh: ndarray = local_model.predict(x_neigh)
    return local_model, x_neigh, y_neigh


def parse():
    parser = argparse.ArgumentParser(description='Check the neighborhood generation for every row of the dataset.')
    parser.add_argument('randomforest', metavar='RF', type=str, help='random forest path')
    parser.add_argument('data', metavar='DATA', type=str, help="dictionary containing training and test sets")
    args_ = parser.parse_args()
    return args_


def check_row(row_id, features, targets, labels, bb, explainer):
    x = features[row_id]
    y = targets[row_id]

    model, x_neigh, y_neigh = __get_local_data(x, y, black_box=bb, num_samples=5000, labels=labels, exp=explainer,
                                               sampling_method="lhs")

    set_lab = set(labels)
    set_y_neigh = set(y_neigh)
    if len(set_lab - set_y_neigh) != 0:
        print(f"Row {row_id}: y_neigh: {set_y_neigh}")


if __name__ == '__main__':
    args = parse()
    rand_for = load_pickle_bz2(args.randomforest)
    bb = SklearnBlackBox(rand_for)
    print("Black Box Loaded")
    # loading the data
    loaded = np.load(args.data, allow_pickle=True)
    try:
        x_train: ndarray = loaded["x_train"]
        y_train: ndarray = loaded["y_train"]
        x_test: ndarray = loaded["x_test"]
        y_test: ndarray = loaded["y_test"]
    except KeyError as e:
        x_train: ndarray = loaded["X_train"]
        y_train: ndarray = loaded["y_train"]
        x_test: ndarray = loaded["X_test"]
        y_test: ndarray = loaded["y_test"]
    print("Data Loaded")

    labels = np.unique(np.concatenate([y_train, y_test])).tolist()
    print(f"Labels Detected: {labels}")

    indices: int = range(len(x_train))

    explainer = LimeTabularExplainer(x_train, random_state=123)

    with Parallel(n_jobs=-1, prefer="processes") as parallel:
        parallel(
            delayed(check_row)(r_id, x_train, y_train, labels, bb, explainer) for r_id in indices
        )
