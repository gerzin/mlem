#!/usr/bin/env python3
import argparse

import numpy as np
import yaml
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.append("../")
sys.path.append("../lime")
from mlem.utilities import load_pickle_bz2

RESOURCES_START_PATH = Path(__file__).parent.parent
EXPERIMENT_FOLDER = Path(__file__).parent / "experiments_results"


def parse_experiment_file(file_path):
    fp = Path(file_path)
    if not fp.is_file():
        print(f"[ERROR] Invalid experiment file: {fp}")
        exit(1)
    elif (suff := fp.suffix.replace(".", "")) not in "yamlymlYAML":
        print(f"[ERROR] expfile must end in [yaml/yml] got {suff}")
        exit(1)
    with open(fp, "r") as f:
        file = f.read()
        yml = yaml.full_load(file)
        return yml


def make_results_folder(folder_name) -> Path:
    results_path = EXPERIMENT_FOLDER / folder_name
    if results_path.exists():
        print(f"[ERROR] A folder named {folder_name} already exists in {EXPERIMENT_FOLDER}")
        exit(1)
    else:
        results_path.mkdir(exist_ok=False)
        print(f"[INFO] {results_path} created")
    return results_path


def load_blackbox(bbdict):
    bb_path = RESOURCES_START_PATH / bbdict["path"]
    if not bb_path.is_file():
        print(f"[ERROR] Can't find file {bb_path}")
        exit(1)

    bb = load_pickle_bz2(bb_path)
    print(f"[INFO] BlackBox correctly loaded from {bb_path}")
    print(f"\t[INFO] BlackBox type: {type(bb).__name__}")
    return bb


def load_blackbox_data(bbdatadict):
    bb_data_path = RESOURCES_START_PATH / bbdatadict['path']
    if not bb_data_path.is_file():
        print(f"[ERROR] Can't find file {bb_data_path}")
        exit(1)
    print("[INFO] Loading blackbox data")
    loaded = np.load(str(bb_data_path))
    x_train = None
    x_test = None
    y_train = None
    y_test = None
    x_attack = None
    y_attack = None
    bb_data_dict = {}

    for k in loaded.keys():
        k_low = k.lower()
        if k_low == "x_train":
            x_train = loaded[k]
            print(f"\t[INFO] x_train loaded: {x_train.shape=}")
        elif k_low == "y_train":
            y_train = loaded[k]
            print(f"\t[INFO] y_train loaded: {y_train.shape=}")
        elif k_low == "x_test":
            x_test = loaded[k]
            print(f"\t[INFO] x_test loaded: {x_test.shape=}")
        elif k_low == "y_test":
            y_test = loaded[k]
            print(f"\t[INFO] y_test loaded: {y_test.shape=}")
        elif k_low in ["x_test_clustered", "x_attack"]:
            assert x_attack is None  # check you don't have both keys
            x_attack = loaded[k]
            print(f"\t[INFO] x_attack loaded with key {k}: {x_attack.shape=}")
        elif k_low in ["y_test_clustered", "y_attack"]:
            assert y_attack is None  # check you don't have both keys
            y_attack = loaded[k]
            print(f"\t[INFO] y_attack loaded with key {k}: {y_attack.shape=}")
    if (x_attack is None) or (y_attack is None):
        print("[ERROR] Couldn't find x_attack or y_attack")
        exit(0)
    print(f"[INFO] Data correctly loaded from {bb_data_path}")

    features_cols, target_col = None, None
    if bbdatadict['columns'] is None:
        print("\t[WARNING] Missing dataset columns from.")
    else:
        features_cols = bbdatadict['columns']["features"]
        target_col = bbdatadict['columns']["target"]
        print(f"\t[INFO] Loaded feature columns: {features_cols}")
        print(f"\t[INFO] Loaded target column: {target_col}")

    for k in ["x_train", "y_train", "x_test", "y_test", "x_attack", "y_attack", "features_cols", "target_col"]:
        bb_data_dict[k] = eval(k)
    return bb_data_dict


def load_experiment_settings(expdict):
    expdata = {}
    print("[INFO] Loading experiment settings")
    # results folder
    expdata['results_path'] = make_results_folder(expdict["results_folder_name"])

    # n_rows
    n_rows = expdict["n_rows"]
    if n_rows is None:
        print("\t[INFO] Loaded n_rows: using all the rows")
    else:
        print(f"\t[INFO] Loaded n_rows: using {n_rows} row{'s' if n_rows > 1 else ''}")
    expdata['n_rows'] = n_rows
    # local_attack_dataset
    local_attack_dataset_path = expdict['local_attack_dataset_path']
    print(f"\t[INFO] local_attack_dataset path: {local_attack_dataset_path}")
    if local_attack_dataset_path is not None:
        ldp = Path(local_attack_dataset_path)
    return expdata


def attack_pipeline(bb_dict, bb_data):
    pass


def main():
    parser = argparse.ArgumentParser(description="Run a mlem experiment contained in a yaml file.")
    parser.add_argument("expfile", help="the YAML file containing the experiment specifications")
    args = parser.parse_args()
    yml = parse_experiment_file(args.expfile)
    print(yml)
    BLACK_BOX = load_blackbox(yml["black_box"])
    BLACK_BOX_DATA = load_blackbox_data(yml['black_box']['data'])
    EXPERIMENT_DATA = load_experiment_settings(yml['experiment'])
    print(EXPERIMENT_DATA)

    # copying the yml file in the experiment results folder so to have a description of the experiment contained in the
    # folder
    with open(EXPERIMENT_DATA['results_path'] / Path(args.expfile).name, "w") as f:
        with open(args.expfile, "r") as g:
            f.write(g.read())
        print("[INFO] Saved copy of experiment file")
    attack_pipeline(BLACK_BOX, BLACK_BOX_DATA)


if __name__ == '__main__':
    main()
