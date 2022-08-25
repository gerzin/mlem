from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def __check_array_size(arr, arr_name):
    if arr.size == 0:
        print(f"[ERROR] {arr_name} has size 0")


def evaluate_attack(train_set, test_set, atk0, atk1, black_box, atk_title: str, ax=None):
    train_x = train_set.drop('Target', axis=1)
    test_x = test_set.drop('Target', axis=1)

    train_predictions = black_box.predict_proba(train_x.to_numpy())
    test_predictions = black_box.predict_proba(test_x.to_numpy())

    train_preds_0 = train_predictions[train_set['Target'] == 0]
    train_preds_1 = train_predictions[train_set['Target'] == 1]

    test_preds_0 = test_predictions[test_set['Target'] == 0]
    test_preds_1 = test_predictions[test_set['Target'] == 1]

    # some checks
    __check_array_size(train_preds_0, "train_preds_0")
    __check_array_size(train_preds_1, "train_preds_1")
    __check_array_size(test_preds_0, "test_preds_0")
    __check_array_size(test_preds_1, "test_preds_1")

    attack_preds_in = np.concatenate([
        atk0.predict(train_preds_0),
        atk1.predict(train_preds_1)
    ])

    attack_preds_out = np.concatenate([
        atk0.predict(test_preds_0),
        atk1.predict(test_preds_1)
    ])
    print(f"{attack_preds_out=}")
    print(
        f'#even = {len([x for x in attack_preds_in if x == "even"])} IN - {len([x for x in attack_preds_out if x == "even"])} OUT')

    in_labels = ['in'] * len(attack_preds_in)
    out_labels = ['out'] * len(attack_preds_out)

    labels = np.concatenate([in_labels, out_labels])
    preds = np.concatenate([attack_preds_in, attack_preds_out])

    even_mask = preds != "even"

    assert len(labels) == len(preds) and len(even_mask) == len(labels)
    assert set(labels) == set(['in', 'out'])
    labels = labels[even_mask]
    preds = preds[even_mask]

    print(atk_title)
    print(classification_report(labels, preds))

    ConfusionMatrixDisplay.from_predictions(labels, preds, ax=ax, cmap='inferno')
    if ax:
        ax.set_title(atk_title)
    else:
        plt.title(atk_title)


def evaluate_attack_model(train_set, test_set, attack_model):
    pass
