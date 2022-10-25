from pathlib import Path
from utils.models import load_pickle_bz2
import numpy as np

def load_attack_models(base_folder_path):
    """ Load the attack models

    Params:
        base_folder_path - path of the experiment_output folder
    Returns:
        attack_models_0, attack_models_1, instances
    
    """
    base_folder = Path(base_folder_path)
    assert base_folder.name == 'experiment_output'
    atk_models_0 = []
    atk_models_1 = []
    atk_models_instances = []
    for p in base_folder.iterdir():
        if p.is_dir():
            attack_0 = p / "attack" / "0" / "model.pkl.bz2"
            attack_1 = p / "attack" / "1" / "model.pkl.bz2"
            instance = p / "instance.npy"
            if attack_0.exists() and attack_1.exists():
                atk_0 = load_pickle_bz2(attack_0)
                atk_1 = load_pickle_bz2(attack_1)
                instance = np.load(instance, allow_pickle=True)
                atk_models_0.append(atk_0)
                atk_models_1.append(atk_1)
                atk_models_instances.append(instance)
    return atk_models_0, atk_models_1, atk_models_instances