import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

def __save_txt(path, txt):
    """Saves a text file on disk.

    Args:
        path (str | Path): Path where to save the string.
        txt (str): String to save.
    """
    path = str(path)
    with open(path, "w") as f:
        f.write(txt)


def __plot_confusion_matrix(y_true, y_predicted, ax):
    ax.grid(False)
    ConfusionMatrixDisplay.from_predictions(y_true, y_predicted, cmap='inferno', ax=ax, colorbar=False)


def evaluate_attack(atk0, atk1, black_box, black_box_data, output_folder=None, split_true_label=False):
    
    # check that black_box data contains the right keys
    for key in ('X_train', 'X_test', 'y_train', 'y_test'):
        if not key in black_box_data.keys():
            raise ValueError(f"Missing key {key} from black_box_data")
    
    # putting the train and test set in a dataframe
    features_train = black_box_data['X_train']
    train_set = pd.DataFrame(black_box_data['X_train'])
    train_set['Target'] = black_box_data['y_train']
    probs_train = black_box.predict_proba(features_train)
    train_set['BlackBoxProb0'] = probs_train[:,0]
    train_set['BlackBoxProb1'] = probs_train[:,1]
    train_set['Position'] = 'in'


    features_test = black_box_data['X_test']
    test_set = pd.DataFrame(black_box_data['X_test'])
    test_set['Target'] = black_box_data['y_test']
    probs_test = black_box.predict_proba(features_test)
    test_set['BlackBoxProb0'] = probs_test[:,0]
    test_set['BlackBoxProb1'] = probs_test[:,1]
    test_set['Position'] = 'out'

    # concatenation of the train and test set
    train_test = pd.concat([train_set, test_set])

    zeroes = train_test[train_test.BlackBoxProb0 >= 0.5].copy() if not split_true_label else train_test[train_test.Target == 0].copy()
    ones = train_test[train_test.BlackBoxProb0 < 0.5].copy() if not split_true_label else train_test[train_test.Target == 1].copy()

    # run the attack
    zeroes['Atk'] = atk0.predict(zeroes[["BlackBoxProb0","BlackBoxProb1"]].to_numpy())
    ones['Atk'] = atk1.predict(ones[["BlackBoxProb0","BlackBoxProb1"]].to_numpy())

    train_test = pd.concat([zeroes, ones])

    train_test = train_test[train_test.Atk != 'even']
    class0 = train_test[train_test["BlackBoxProb0"] >= 0.5]
    class1 = train_test[train_test["BlackBoxProb0"] < 0.5]

    report_full = classification_report(train_test.Position, train_test.Atk)
    report_0 = classification_report(class0.Position, class0.Atk)
    report_1 = classification_report(class1.Position, class1.Atk)

    
    print("report full")
    print(report_full)
    print("\nClass 0")
    print(report_0)
    print("\nClass 1")
    print(report_1)



    if output_folder:
        __save_txt(output_folder / "classification_report_full.txt", report_full)
    
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20,18))

    axs_rav = axs.ravel()
    
    __plot_confusion_matrix(train_test.Position, train_test.Atk, ax=axs_rav[0])
    __plot_confusion_matrix(class0.Position, class0.Atk, ax=axs_rav[1])
    __plot_confusion_matrix(class1.Position, class1.Atk, ax=axs_rav[2])

    axs[0].set_title("Full")
    axs[1].set_title("Class 0")
    axs[2].set_title("Class 1")

    plt.tight_layout()

def attack_bb_dataset(atk_0, atk_1, black_box, bb_data):
    
    # extracting train and test set of the BB
    BB_TRAIN = pd.DataFrame(bb_data['X_train'])
    BB_TRAIN['Target'] = bb_data['y_train']

    BB_TEST = pd.DataFrame(bb_data['X_test'])
    BB_TEST['Target'] = bb_data['y_test']
    
    # for each set separate the elements belonging to class 0 from the ones of class 1
    BB_TEST_0 = BB_TEST[BB_TEST.Target == 0]
    BB_TEST_1 = BB_TEST[BB_TEST.Target == 1]

    BB_TRAIN_0 = BB_TRAIN[BB_TRAIN.Target == 0]
    BB_TRAIN_1 = BB_TRAIN[BB_TRAIN.Target == 1]
    
    # use the black box to compute the probabilities
    BB_TEST_0 = pd.DataFrame(black_box.predict_proba(BB_TEST_0.drop('Target', axis=1).to_numpy()))
    BB_TEST_1 = pd.DataFrame(black_box.predict_proba(BB_TEST_1.drop('Target', axis=1).to_numpy()))
    BB_TRAIN_0 = pd.DataFrame(black_box.predict_proba(BB_TRAIN_0.drop('Target', axis=1).to_numpy()))
    BB_TRAIN_1 = pd.DataFrame(black_box.predict_proba(BB_TRAIN_1.drop('Target', axis=1).to_numpy()))
    
    # use the attack models
    BB_TEST_0['ATK'] = atk_0.predict(BB_TEST_0.to_numpy())
    BB_TEST_1['ATK'] = atk_1.predict(BB_TEST_1.to_numpy())
    
    BB_TRAIN_0['ATK'] = atk_0.predict(BB_TRAIN_0.to_numpy())
    BB_TRAIN_1['ATK'] = atk_1.predict(BB_TRAIN_1.to_numpy())
    
    # assign the true label to each element so that it can be easily compared to the one from the attack model
    BB_TEST_0['Y'] = 'out'
    BB_TEST_1['Y'] = 'out'
    BB_TRAIN_0['Y'] = 'in'
    BB_TRAIN_1['Y'] = 'in'
    
    # concatenate everything
    train_test = pd.concat([BB_TEST_0, BB_TEST_1,BB_TRAIN_0, BB_TRAIN_1])
    return train_test

def evaluate_attack_distances(atk0, atk1, black_box, black_box_data, output_folder=None, split_true_label=False):
    """Like evaluate_attack but used for ensembles that exploit the distance from the point to explain.
    """
    # check that black_box data contains the right keys
    for key in ('X_train', 'X_test', 'y_train', 'y_test'):
        if not key in black_box_data.keys():
            raise ValueError(f"Missing key {key} from black_box_data")
    
    # putting the train and test set in a dataframe
    features_train = black_box_data['X_train']
    train_set = pd.DataFrame(black_box_data['X_train'])
    train_set['Target'] = black_box_data['y_train']
    probs_train = black_box.predict_proba(features_train)
    train_set['BlackBoxProb0'] = probs_train[:,0]
    train_set['BlackBoxProb1'] = probs_train[:,1]
    train_set['Position'] = 'in'


    features_test = black_box_data['X_test']
    test_set = pd.DataFrame(black_box_data['X_test'])
    test_set['Target'] = black_box_data['y_test']
    probs_test = black_box.predict_proba(features_test)
    test_set['BlackBoxProb0'] = probs_test[:,0]
    test_set['BlackBoxProb1'] = probs_test[:,1]
    test_set['Position'] = 'out'

    # concatenation of the train and test set
    train_test = pd.concat([train_set, test_set])

    zeroes = train_test[train_test.BlackBoxProb0 >= 0.5].copy() if not split_true_label else train_test[train_test.Target == 0].copy()
    ones = train_test[train_test.BlackBoxProb0 < 0.5].copy() if not split_true_label else train_test[train_test.Target == 1].copy()


    zeroes['Atk'] = atk0.predict(zeroes[["BlackBoxProb0","BlackBoxProb1"]].to_numpy(), zeroes.drop(labels=["Target", "BlackBoxProb0","BlackBoxProb1", "Position"], axis=1).to_numpy())
    ones['Atk'] = atk1.predict(ones[["BlackBoxProb0","BlackBoxProb1"]].to_numpy(), ones.drop(labels=["Target", "BlackBoxProb0","BlackBoxProb1", "Position"], axis=1).to_numpy())

    train_test = pd.concat([zeroes, ones])

    train_test = train_test[train_test.Atk != 'even']
    class0 = train_test[train_test["BlackBoxProb0"] >= 0.5]
    class1 = train_test[train_test["BlackBoxProb0"] < 0.5]

    report_full = classification_report(train_test.Position, train_test.Atk)
    report_0 = classification_report(class0.Position, class0.Atk)
    report_1 = classification_report(class1.Position, class1.Atk)

    
    print("report full")
    print(report_full)
    print("\nClass 0")
    print(report_0)
    print("\nClass 1")
    print(report_1)



    if output_folder:
        __save_txt(output_folder / "classification_report_full.txt", report_full)
    
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20,18))

    axs_rav = axs.ravel()
    
    __plot_confusion_matrix(train_test.Position, train_test.Atk, ax=axs_rav[0])
    __plot_confusion_matrix(class0.Position, class0.Atk, ax=axs_rav[1])
    __plot_confusion_matrix(class1.Position, class1.Atk, ax=axs_rav[2])

    axs[0].set_title("Full")
    axs[1].set_title("Class 0")
    axs[2].set_title("Class 1")

    plt.tight_layout()