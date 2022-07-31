"""
This module contains the pipeline used to perform an attack on a model.
"""
import os
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray, delete
from numpy.lib.npyio import savez_compressed
from pandas.core.frame import DataFrame
from lime.lime_tabular import LimeTabularExplainer
from mlem.attack_models import AttackModelsManager, AttackStrategy
from mlem.ensemble import EnsembleClassifier
from mlem.enumerators import SamplingTechnique
from mlem.black_box import BlackBox
from mlem.explainer import LoreDTLoader
from mlem.shadow_models import ShadowModelsManager
from mlem.utilities import create_attack_dataset, save_pickle_bz2, save_txt, create_random_forest, \
    create_dataset_for_attack, oversample
import time

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def __generate_neighborhood(
        instance: ndarray,
        explainer: LimeTabularExplainer,
        num_samples: int,
        **kwargs,
) -> ndarray:
    """
    Generate the neighborhood of an instance.

    Args:
        instance: instance around which to generate the neighborhood.
        explainer: explainer used to generate the neighborhood. The supported ones are: LimeTabularExplainer
        num_samples: number of neighbors to generate.
    Keyword Args:
        sampling_method: (LimeTabularExplainer)
    Returns:
        neighborhood - neighbors of the instance
    """
    if type(explainer) is LimeTabularExplainer:
        # Generates the neighborhood

        sampling_method = kwargs.get('sampling_method')
        if sampling_method is None:
            raise ValueError("Lime needs a sampling method")

        _, neighborhood = explainer.data_inverse(
            instance, num_samples, sampling_method.value
        )
        # Deletes the first row
        neighborhood = delete(neighborhood, 0, axis=0)
        return neighborhood
    else:
        raise TypeError("Unsupported explainer type")


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


def perform_attack_pipeline(
        idx: int,
        x: ndarray,
        y: ndarray,
        labels: List[Any],
        black_box: BlackBox,
        results_path: str,
        explainer: Union[LimeTabularExplainer, LoreDTLoader],
        explainer_sampling: SamplingTechnique,
        neighborhood_sampling: SamplingTechnique,
        attack_full: DataFrame,
        num_samples: int,
        num_shadow_models: int,
        test_size: float,
        random_state: int,
        local_attack_dataset: ndarray = None,
        model_creator_fn=create_random_forest,
        attack_strategy: AttackStrategy = AttackStrategy.ONE_PER_LABEL,
        **kwargs
):
    """
    Execute the MIA Attack with a Local Explainer model on an instance.



    Args:

        idx: index of the row used for the attack.
        x: row of the train dataset
        y: label of x
        labels: list containing all the labels of the train dataset.
        black_box: black box to attack.
        results_path: path where to store the results.
        explainer:
        explainer_sampling:
        neighborhood_sampling:
        attack_full: full attack dataset consisting of (probab. vector, true label, in/out label)
        num_samples:
        num_shadow_models:
        test_size:
        random_state: seed of random number generators.
        local_attack_dataset: dataset to label with the local model and use for the creation of the shadow models. (default None)
        attack_strategy: how to build the attack models, default one for each label
    Keyword Args:
        categorical_mask (bool list): The entry is True if the corresponding feature is categorical else False. (by default it considers all features as numerical)

    Returns:

    """
    print(f"Attacking index {idx} using {attack_strategy.name} strategy")
    start_time = time.time()

    if type(explainer) is LimeTabularExplainer:
        # Creates a local explainer with a neighborhood
        local_model, x_neigh, y_neigh = __get_local_data(
            x, y, explainer, black_box, explainer_sampling, num_samples, labels
        )

        if set(labels) != set(y_neigh):
            print(
                f"WARNING: Neighborhood of index {idx} doesn't contain all labels. Missing: {set(labels) - set(y_neigh)}")

    elif type(explainer) is LoreDTLoader:
        # load the local model
        local_model = explainer.load(index=idx)

    # categorical mask
    categorical_mask = kwargs.get("categorical_mask", [False for _ in range(len(x))])
    assert all([type(x) is bool for x in categorical_mask])

    # Path of the current attacked object
    path: str = f"{results_path}/{idx}"
    # Path where to save the black box and its data
    black_box_path: str = f"{path}/black_box"

    # If needed extracts the attack model
    x_attack: ndarray = None
    y_attack: ndarray = None
    if local_attack_dataset is not None:
        x_attack = local_attack_dataset
        y_attack = local_model.predict(x_attack)

        x_attack = create_dataset_for_attack(x_neigh, black_box, local_attack_dataset, 5000)
        y_attack = local_model.predict(x_attack)

    elif neighborhood_sampling == SamplingTechnique.SAME:
        x_attack = x_neigh
        y_attack = y_neigh

    else:
        x_attack = __generate_neighborhood(
            instance=x, explainer=explainer, num_samples=num_samples, sampling_method=neighborhood_sampling
        )
        y_attack = black_box.predict(x_attack)
        assert len(np.unique(y_attack)) > 1
        _ones_distr = sum([y == 1 for y in y_attack]) / len(y_attack)
        print(f"GENERATED = {1 - _ones_distr} {_ones_distr}%")
        x_attack, y_attack = oversample(x_attack, y_attack, categorical_mask)

        _ones_distr = sum([y == 1 for y in y_attack]) / len(y_attack)
        print(f"GENERATED AFTER OVERSAMPLING = {1 - _ones_distr} {_ones_distr}%")

    # TODO controllare queste due righe
    # Prediction probability on neighborhood
    y_prob: ndarray = black_box.predict_proba(x_attack)
    # Attack dataset created on the neighborhood #qui solo (y_prob, y_attack, "in")
    neighborhood_data: DataFrame = create_attack_dataset(y_prob, y_attack)

    # Creates the shadow models path
    os.makedirs(black_box_path, exist_ok=True)
    # Saves the local model on disk
    # print(f"saving local model for {idx} in {black_box_path}/model.pkl.bz2")
    save_pickle_bz2(f"{black_box_path}/model.pkl.bz2", local_model)

    if type(explainer) is LimeTabularExplainer:
        # Saves the neighborhood-generated data on disk
        savez_compressed(f"{black_box_path}/data", x=x_neigh, y=y_neigh)

    # Creates a number of shadow models to imitate the local model
    shadow_models = ShadowModelsManager(
        n_models=num_shadow_models,
        results_path=f"{path}/shadow",
        test_size=test_size,
        random_state=random_state,
        model_creator_fn=model_creator_fn,
        categorical_mask=categorical_mask
    )
    # Fits the shadow models to imitate the black boxes.

    shadow_models.fit(x_attack, y_attack)

    # Extracts the attack dataset of the shadow model for the attack models
    attack_dataset: DataFrame = shadow_models.get_attack_dataset()

    # Creates a number of attack models that infer the relation between probability and belonging
    attack_models = AttackModelsManager(
        results_path=f"{path}/attack", model_creator_fn=model_creator_fn, attack_strategy=attack_strategy
    )

    # Fits the attack models
    attack_models.fit(attack_dataset)

    # Checks if the record (in the training set) is recognized as "in" # how do we know which elements of x_attack were in the training set?
    pred_class: str = attack_models.predict(y_prob, y)
    save_txt(
        f"{path}/attack/x_is_in.txt",
        f"x = {x}\ny = {y}\ny_prob={y_prob}\npredicted = {pred_class}"
    )
    # Performs the test on the attack dataset created from the neighborhood
    # attack_models.audit(neighborhood_data, "neighborhood") # commented out, due to division by zero warnings

    # Performs the test on the full attack dataset
    attack_models.audit(attack_full.drop(index=idx), "full")
    print(f"attack on {idx} lasted {get_duration_(start_time)}")


def get_duration_(start):
    """

    Args:
        start:

    Returns:
        string of the format "{minutes}:{seconds}m"
    """
    n = time.time() - start
    m = int(n // 60)
    s = int(n - (m * 60))
    return f"{m}:{s}m"
