from pathlib import Path
import bz2
import numpy as np
import pickle
import logging

logging.basicConfig(level=logging.INFO)


def get_attack_mod(index: int, base_folder, targets=[0, 1]):
    """
    Args:
        index: row of the dataset on which the attack model was built
        base_folder: path up to the index
        targets (int | List(int)) - targets for the attack model.
    Returns:
        List containing the attack models (or model) for that particular index
    """
    if type(targets) is int:
        targets = [targets]
    loaded_models = []
    for t in targets:
        path = Path(base_folder) / f"{index}" / "attack" / f"{t}" / "model.pkl.bz2"
        with bz2.BZ2File(path) as f:
            data = pickle.load(f)
            loaded_models.append(data)
        logging.debug(f"Loading attack model from {path}")
    return loaded_models


def get_attack_model_data(index: int, base_folder, targets=[0, 1]):
    """
    Load the attack models data. The returned objects have the keys (x_train x_test y_train y_test)
    Args:
        index: row of the dataset on which the attack model was built
        base_folder: path up to the index
        targets (int | List(int)) - targets for the attack model.
    Returns:
        List containing the attack models' data for that particular index (label0, label1)
    """
    if type(targets) is int:
        targets = [targets]
    loaded_data = []
    for t in targets:
        path = Path(base_folder) / f"{index}" / "attack" / f"{t}" / "data.npz"
        data = np.load(path, allow_pickle=True)
        loaded_data.append(data)
    return loaded_data


def get_local_model(index: int, base_folder):
    """
    Get the local model built on a row.
    Args:
        index: row of the dataset on which the local model was built.
        base_folder: base path of the local model up to part before the row number
    Returns:
        Local model
    """
    path = Path(base_folder) / f"{index}" / "black_box" / "model.pkl.bz2"
    with bz2.BZ2File(path) as lm:
        local_model = pickle.load(lm)
    return local_model


def get_local_model_data(index: int, base_folder):
    """
    Get the data of a local model.
    Args:
        index - row of the dataset on which the local model was built.
    Returns:
        Structure with keys x and y
    """
    path = Path(base_folder) / f"{index}" / "black_box" / "data.npz"
    loaded = np.load(str(path), allow_pickle=True)
    return loaded


def create_ensembles(base_folder, ensemble_class, verbose=False):
    """

    Args:
        base_folder: path to the folder containing the subfolders'
        ensemble_class: HardVotingClassifier or SoftVotingClassifier

    Returns:
        (ensemble0, ensemble1): ensemble for the label 0 and 1
    """
    ATTACK_0, ATTACK_1 = [], []
    indices = [int(i.stem) for i in Path(base_folder).iterdir() if i.is_dir() and i.stem.isdigit()]
    indices.sort()
    assert len(indices) > 0

    if verbose:
        print(f"Loaded {len(indices)} rows")

    for index in indices:
        atk0, atk1 = get_attack_mod(index, base_folder)
        ATTACK_0.append(atk0)
        ATTACK_1.append(atk1)

    ensemble_0 = ensemble_class(classifiers=ATTACK_0)
    ensemble_1 = ensemble_class(classifiers=ATTACK_1)

    return ensemble_0, ensemble_1


def create_ensamble_attack_dataset(base_folder, kind):
    """

    Returns:
        (Features0, label0), (Features1, label1) - The feature and labels for the attack models trained to predict membership on label 0 and 1

    """
    assert kind.lower() in ("train", "test")

    indices = [int(i.stem) for i in Path(base_folder).iterdir() if i.is_dir() and i.stem.isdigit()]
    indices.sort()
    assert len(indices) > 0

    DATA_0_X = []
    DATA_0_Y = []
    DATA_1_X = []
    DATA_1_Y = []

    key_x = f"x_{kind}"
    key_y = f"y_{kind}"

    for index in indices:
        data0, data1 = get_attack_model_data(index, base_folder)

        DATA_0_X.append(data0[key_x])
        DATA_0_Y.append(data0[key_y])

        DATA_1_X.append(data1[key_x])
        DATA_1_Y.append(data1[key_y])

    X_0 = np.concatenate(DATA_0_X)
    Y_0 = np.concatenate(DATA_0_Y)
    X_1 = np.concatenate(DATA_1_X)
    Y_1 = np.concatenate(DATA_1_Y)

    return (X_0, Y_0), (X_1, Y_1)
