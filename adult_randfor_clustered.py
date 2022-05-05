#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from typing import Any, List
from numpy.core.shape_base import vstack
from numpy.lib.arraysetops import unique
from pandas.core.frame import DataFrame

from mlem.utilities import create_attack_dataset, load_pickle_bz2

# Adds LIME to the system
sys.path.append("./lime")

from pandas import read_pickle
import torch
from mlem.enumerators import BlackBoxType, ExplainerType, SamplingTechnique

from numpy import concatenate, ndarray, load
from typer import echo, run
from lime.lime_tabular import LimeTabularExplainer

from mlem.black_box import BlackBox, PyTorchBlackBox, SklearnBlackBox
from mlem.neural_black_box import Net

from joblib import Parallel, delayed, cpu_count

from mlem.attack_pipeline import perform_attack_pipeline
import pdb

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def __full_attack_dataset(
        black_box: BlackBox,
        x_train: ndarray,
        y_train: ndarray,
        x_test: ndarray,
        y_test: ndarray,
) -> DataFrame:
    """Creates an attack dataset once and for all, based on the full training set.

    Args:
        black_box (BlackBox): Black box used to perform the prediction.
        x_train (ndarray): Input training set.
        y_train (ndarray): Training target label.
        x_test (ndarray): Input training set.
        y_test (ndarray): Input test set.

    Returns:
        DataFrame: Attack dataset for the full input dataset in the black box.
    """
    # Prediction probabilities for the training set
    y_prob_train: ndarray = black_box.predict_proba(x_train)
    # Prediction probabilities for the test set
    y_prob_test: ndarray = black_box.predict_proba(x_test)
    # Attack dataframe created accordingly
    attack: DataFrame = create_attack_dataset(
        y_prob_train, y_train, y_prob_test, y_test
    )
    return attack


def main(
        black_box_type: BlackBoxType,
        black_box_path: str,
        data_path: str,
        results_path: str = "./results",
        explainer_type: ExplainerType = "lime",
        explainer_sampling: SamplingTechnique = "gaussian",
        neighborhood_sampling: SamplingTechnique = "same",
        num_samples: int = 5000,
        num_shadow_models: int = 4,
        test_size: float = 0.2,
        random_state: int = 42,
        n_jobs: int = -1,
        n_rows: int = -1,
):
    """Starts a new experimental suite of MLEM.

    Args:\n
        black_box_type (BlackBoxType): Kind of black box to use.\n
        black_box_path (str): Path of the Pickle file where to pick the black box classifier.\n
        data_path (str): Path of the CSV input data.\n # TODO CSV??? or npz with (X_train, X_test, y_train, y_test)
        results_path (str, optional): Path where to save the intermediate results. Defaults to "./results".\n
        explainer_type (ExplainerType, optional): Local explainer to use. Defaults to "lime".\n
        explainer_sampling (SamplingTechnique, optional): Type of sampling performed by the local Explainer to explain a local result. Defaults to "gaussian".\n
        neighborhood_sampling (SamplingTechnique, optional): Type of sampling performed by the local Explainer to perform the MIA. Defaults to "same" (same dataset as before).\n
        num_samples (int, optional): Number of samples for the neighborhood generation. Defaults to 5000.\n
        num_shadow_models (int, optional): Number of shadow models to use in order to mimic the black box. Defaults to 4.\n
        test_size (float, optional): Size of test (in proportion) to extract the data. Defaults to 0.2.\n
        random_state (int, optional): Seed of random number generators. Defaults to 42.\n
        n_jobs (int, optional): Number of jobs used by JobLib to parallelize the works. Defaults to -1 (all the available cores).\n
        n_rows (int, optional): Number of rows of the dataset on which to perform the MIA. Defaults to -1 (all the rows).\n
    """
    echo("MLEM: MIA (Membership Inference Attack) of Local Explanation Methods")

    if Path(results_path).exists():
        print(f"The results_path {results_path} already exists. Continue and overwrite? [y/n]")
        resp = input("> ")
        if not resp.lower() in "yes":
            exit(1)

    # Load the black box model
    black_box: BlackBox = None
    if black_box_type == BlackBoxType.NN:
        net = Net()
        net.load_state_dict(torch.load(black_box_path))
        black_box = PyTorchBlackBox(net)
    elif black_box_type == BlackBoxType.RF:
        model = load_pickle_bz2(black_box_path)
        black_box = SklearnBlackBox(model)
    else:
        echo("Not a valid black box", err=True)
        exit(1)
    echo("Black box model correctly read")
    # Set the sampling method.
    if explainer_sampling == SamplingTechnique.SAME:
        echo("Sampling of the explainer has to be either 'gaussian' or 'lhs'", err=True)
        exit(1)
    # Load the input dataset
    loaded = load(data_path, allow_pickle=True)
    try:
        x_train: ndarray = loaded["x_train"]
        y_train: ndarray = loaded["y_train"]
        x_test: ndarray = loaded["x_test"]
        y_test: ndarray = loaded["y_test"]
        x_test_clustered: ndarray = loaded["x_test_clustered"]
        y_test_clustered: ndarray = loaded["y_test_clustered"]
    except KeyError as e:
        x_train: ndarray = loaded["X_train"]
        y_train: ndarray = loaded["y_train"]
        x_test: ndarray = loaded["X_test"]
        y_test: ndarray = loaded["y_test"]
        x_test_clustered: ndarray = loaded["x_test_clustered"]
        y_test_clustered: ndarray = loaded["y_test_clustered"]
    # pdb.set_trace()
    # List of target labels
    labels: List[Any] = unique(concatenate([y_train, y_test])).tolist()
    labels = [int(x) for x in labels]
    # Target labels
    echo("Input data correctly read from disk")

    # Attack dataset used in the audit
    attack_full: DataFrame = __full_attack_dataset(
        black_box, x_train, y_train, x_test, y_test
    )

    # Creates the result folder if it does not exist
    os.makedirs(results_path, exist_ok=True)
    echo("Output folder correctly created")

    # Local explainer
    explainer = None
    if explainer_type == ExplainerType.LIME:
        explainer = LimeTabularExplainer(x_train, random_state=random_state)
    else:
        echo("Not a valid local explainer.", err=True)
        exit(1)
    echo("Explainer correctly created")

    # Specific path for this experimental setting
    path: str = (
        f"{results_path}/{explainer_sampling.value}/{neighborhood_sampling.value}"
    )
    # Creates the path if it does not exist
    os.makedirs(path, exist_ok=True)
    echo(f"Results are going to be saved in {path}")

    # Indices of the rows of the train dataset
    indices: int = range(len(x_test_clustered))
    # Batch size
    batch_size: int = len(x_train) // cpu_count()

    if n_rows == -1:
        n_rows = len(x_train)
        echo(f"Starting MIA for each row. Tot rows = {len(x_train)}")

    else:
        echo(f"Starting MIA for {n_rows} row{'s' if n_rows != 1 else ''}. Tot rows = {len(x_train)}")

    echo(f"Starting Parallel with {n_jobs=} and {batch_size=}")
    with Parallel(n_jobs=n_jobs, prefer="processes", batch_size=batch_size) as parallel:
        # For each row of the matrix perform the MIA
        parallel(
            delayed(perform_attack_pipeline)(
                idx,
                x_row,
                y_row,
                labels,
                black_box,
                path,
                explainer,
                explainer_sampling,
                neighborhood_sampling,
                attack_full,
                num_samples,
                num_shadow_models,
                test_size,
                random_state,
            )
            for idx, x_row, y_row in zip(indices, x_test_clustered, y_test_clustered)
        )
    echo("Experiments are concluded, kudos from MLEM!")


if __name__ == "__main__":
    run(main)
