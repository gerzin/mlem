import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

def evaluate_attack(atk0, atk1, black_box, black_box_data, output_folder=None):
    """_summary_

    Args:
        atk0 (_type_): _description_
        atk1 (_type_): _description_
        black_box (_type_): _description_
        black_box_train (_type_): _description_
        black_box_test (_type_): _description_
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

    train_test = pd.concat([train_set, test_set])

    zeroes = train_test[train_test.BlackBoxProb0 > 0.5].copy()
    ones = train_test[train_test.BlackBoxProb0 < 0.5].copy()

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
        save_txt(output_folder / "classification_report_full.txt", report_full)
    
    fig, axs = plt.subplots(ncols=3, figsize=(20,18))
    
    ConfusionMatrixDisplay.from_predictions(train_test.Position, train_test.Atk, cmap='inferno', ax=axs.ravel()[0])
    ConfusionMatrixDisplay.from_predictions(class0.Position, class0.Atk, cmap='inferno', ax=axs.ravel()[1])
    ConfusionMatrixDisplay.from_predictions(class1.Position, class1.Atk, cmap='inferno', ax=axs.ravel()[2])

    axs[0].set_title("Full")
    axs[1].set_title("Class 0")
    axs[2].set_title("Class 1")

    plt.tight_layout()