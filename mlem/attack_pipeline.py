"""
This module contains the pipeline used to perform an attack on a model.
"""
import os
from typing import Any, List, Sequence, Tuple
from numpy import ndarray, delete
from numpy.lib.npyio import savez_compressed
from pandas.core.frame import DataFrame
from lime.lime_tabular import LimeTabularExplainer
from mlem.attack_models import AttackModelsManager
from mlem.ensemble import EnsembleClassifier
from mlem.enumerators import SamplingTechnique
from mlem.black_box import BlackBox
from mlem.shadow_models import ShadowModelsManager
from mlem.utilities import create_attack_dataset, save_pickle, save_txt

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def __generate_neighborhood(
    instance: ndarray,
    explainer: LimeTabularExplainer,
    num_samples: int,
    sampling_method: SamplingTechnique,
) -> ndarray:
    """
    Generate the neighborhood of an instance.
    Args:
        instance:
        explainer:
        num_samples:
        sampling_method:

    Returns:

    """
    # Generates the neighborhood
    _, neighborhood = explainer.data_inverse(
        instance, num_samples, sampling_method.value
    )
    # Deletes the first row
    neighborhood = delete(neighborhood, 0, axis=0)
    return neighborhood


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
    id: int,
    x: ndarray,
    y: ndarray,
    labels: List[Any],
    black_box: BlackBox,
    results_path: str,
    explainer: LimeTabularExplainer,
    explainer_sampling: SamplingTechnique,
    neighborhood_sampling: SamplingTechnique,
    attack_full: DataFrame,
    num_samples: int,
    num_shadow_models: int,
    test_size: float,
    random_state: int,
):
    """
    Performs
    Args:
        id: id of what
        x:
        y:
        labels:
        black_box:
        results_path:
        explainer:
        explainer_sampling:
        neighborhood_sampling:
        attack_full:
        num_samples:
        num_shadow_models:
        test_size:
        random_state:

    Returns:

    """
    logger.info("Start Attack Pipeline")
    # Creates a local explainer with a neighborhood
    local_model, x_neigh, y_neigh = __get_local_data(
        x, y, explainer, black_box, explainer_sampling, num_samples, labels
    )
    logger.debug("Done get local data")
    # Path of the current attacked object
    path: str = f"{results_path}/{id}"
    # Path where to save the black box and its data
    black_box_path: str = f"{path}/black_box"
    # If needed extracts the attack model
    x_attack: ndarray = None
    y_attack: ndarray = None
    if neighborhood_sampling == SamplingTechnique.SAME:
        x_attack = x_neigh
        y_attack = y_neigh
    else:
        x_attack = __generate_neighborhood(
            x, explainer, num_samples, neighborhood_sampling
        )
        y_attack = black_box.predict(x_attack)
    # Prediction probability
    y_prob: ndarray = black_box.predict_proba(x_attack)
    # Attack dataset created on the neighborhood
    neighborhood_data: DataFrame = create_attack_dataset(y_prob, y_attack)
    # Creates the shadow models path
    os.makedirs(black_box_path, exist_ok=True)
    # Saves the local model on disk
    save_pickle(f"{black_box_path}/model.pkl.bz2", local_model)
    # Saves the neighborhood-generated data on disk
    savez_compressed(f"{black_box_path}/data", x=x_neigh, y=y_neigh)

    # Creates a number of shadow models to imitate the local model
    shadow_models = ShadowModelsManager(
        n_models=num_shadow_models,
        results_path=f"{path}/shadow",
        test_size=test_size,
        random_state=random_state,
    )
    # Fits the shadow models to imitate the black boxes.

    shadow_models.fit(x_attack, y_attack)

    # Extracts the attack dataset of the shadow model for the attack models
    attack_dataset: DataFrame = shadow_models.get_attack_dataset()

    # Creates a number of attack models that infer the relation between probability and belonging
    attack_models = AttackModelsManager(
        results_path=f"{path}/attack", random_state=random_state
    )

    # Fits the attack models
    attack_models.fit(attack_dataset)
    # Checks if the record (in the training set) is recognized as "in"
    pred_class: str = attack_models.predict(y_prob, y)
    save_txt(
        f"{path}/attack/x_is_in.txt",
        f"x = {x}\ny = {y}\ny_prob={y_prob}\npredicted = {pred_class}"
    )
    # Performs the test on the attack dataset created from the neighborhood
    attack_models.audit(neighborhood_data, "neighborhood")
    # Performs the test on the full attack dataset
    attack_models.audit(attack_full.drop(index=id), "full")
