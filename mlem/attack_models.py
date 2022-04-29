"""
This module contains a class which given a dataset of (pred. vectors, true labels, in/out) records,
creates, trains and evaluates, #true labels Attack Models. Each Attack Model is a RandomForest.
"""
import os
from typing import Dict, List, Tuple
from numpy import concatenate, ndarray
from numpy.lib.npyio import savez_compressed
from pandas.core.frame import DataFrame

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.base import ClassifierMixin
from typing import Callable
from mlem.utilities import create_random_forest, save_pickle_bz2, save_txt


class AttackModelsManager:
    """Class that handles a number of attack models."""

    def __init__(self, results_path: str, model_creator_fn: Callable, random_state: int) -> None:
        """Creates a new manager of various attack models.

        Args:
            results_path (str): String where to save the documents.
            model_creator_fn ( Callable ): function that takes as input features and targets and returns a classifier.
            random_state (int): Random state used in randomization.
        """
        self.results_path = results_path
        self.model_creator = model_creator_fn
        self.random_state = random_state
        # List of attack models
        self.attack_models: Dict[ClassifierMixin] = {}

    def fit(self, attack_dataset: DataFrame) -> None:
        """Fits the attack datasets to predict if the record is inside or outside.
        Contextually it performs the test_single.

        Args:
            attack_dataset (DataFrame): Attack dataset of the form (prob, label, "in"/"out") created by the shadow models.
        """
        # TODO should find a way to parallelize this loop

        # for each label (true class) get the respective dataset
        for label, df in attack_dataset.groupby("label", sort=False):
            # Input data
            x: ndarray = df.drop(columns=["label", "inout"]).values
            y: ndarray = df["inout"].values
            # Path of the current attack model
            path_label_attack: str = f"{self.results_path}/{label}"
            # Creates the path if it does not exist
            os.makedirs(path_label_attack, exist_ok=True)
            # Splits the dataset into train and test
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, random_state=self.random_state
            )
            # Saves the input data
            savez_compressed(
                f"{path_label_attack}/data",
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
            )
            # Attack model used to fit data
            attack_model: ClassifierMixin = self.model_creator(x_train, y_train)
            # Saves the attack model
            save_pickle_bz2(f"{path_label_attack}/model.pkl.bz2", attack_model)
            # Prediction of the model based on data
            y_pred_train: ndarray = attack_model.predict(x_train)
            y_pred_test: ndarray = attack_model.predict(x_test)
            # Report of the classification
            report_train: str = classification_report(y_train, y_pred_train)
            report_test: str = classification_report(y_test, y_pred_test)
            # Saves the report
            save_txt(f"{path_label_attack}/report_train.txt", report_train)
            save_txt(f"{path_label_attack}/report_test.txt", report_test)
            # Saves the model
            self.attack_models[label] = attack_model

    def predict(self, x: ndarray, label: int) -> ndarray:
        """Predicts a new data with one of the attack models.

        Args:
            x (ndarray): Input data.
            label (int): Corresponding label used to retrieve the right attack model.

        Returns:
            ndarray: Output value.
        """
        return self.attack_models[label].predict(x)

    def __test_single(self, data: DataFrame, name: str) -> None:
        """Tests each attack model separately and then concatenates the result.

         It uses
        Notes:
            This method creates files with the classification_report.

        Args:
            data (DataFrame): Data with prediction probability, label and "inout" label.
            name (str): Name of the audited dataset.
        """
        # List of true outputs
        y_true_list: List[ndarray] = []
        # List of predicted outputs
        y_pred_list: List[ndarray] = []
        # Iterates through all the labels
        for label, label_df in data.groupby("label", sort=False):
            # Path of the current attack model
            path: str = f"{self.results_path}/{label}"
            # Get the model for the specific label
            model: ClassifierMixin = self.attack_models[label]
            # Input data
            x: ndarray = label_df.drop(columns=["label", "inout"]).values
            # Label to be predicted, "in" or "out"
            y_true: ndarray = label_df["inout"].values
            # Prediction of the model
            y_pred: ndarray = model.predict(x)
            # create the Classification report and save it on a file.
            report: str = classification_report(y_true, y_pred)
            save_txt(f"{path}/test_single_{name}.txt", report)
            # Updates the list
            y_true_list.append(y_true)
            y_pred_list.append(y_pred)
        # Concatenates the predictions to get an overall report and save it on a file.
        y_true: ndarray = concatenate(y_true_list)
        y_pred: ndarray = concatenate(y_pred_list)
        report: str = classification_report(y_true, y_pred)
        save_txt(f"{self.results_path}/test_single_concat_{name}.txt", report)

    def __test_all(self, data: DataFrame,
                   name: str):  # TODO scrivere che test fa ed effetti collaterali (crea dei file)
        """Tests each attack model on the full attack dataset.

        Args:
            data (DataFrame): Data with prediction probability, label and "inout" label.
            name (str): Name of the audited dataset.
        """
        # Input data
        x: ndarray = data.drop(columns=["label", "inout"]).values
        # Target label
        y: ndarray = data["inout"].values
        # Predictions for each model
        predictions: List[ndarray] = [
            model.predict(x) for model in self.attack_models.values()
        ]
        # Maximum prediction
        max_predictions: List[ndarray] = [
            max(p, key=p.count) for p in zip(*predictions)
        ]
        # Classification report of this prediction
        report: str = classification_report(y, max_predictions)
        # Saves the prediction on disk
        save_txt(f"{self.results_path}/test_all_{name}.txt", report)

    def audit(self, data: DataFrame, name: str) -> None:
        """Tests the attack model with respect to a dataframe created with the create_attack_dataset function.

        Args:
            data (DataFrame): Data with prediction, label and "inout" target label.
            name (str): Name of the dataset.
        """
        self.__test_single(data, name)
        self.__test_all(data, name)
