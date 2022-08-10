import os
from typing import List, Tuple, Callable
import numpy as np
from numpy import ndarray, savez_compressed

from pandas import DataFrame, concat
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE, SMOTENC

from mlem.utilities import (
    create_attack_dataset,
    create_random_forest,
    save_pickle_bz2,
    save_txt, frequencies, minority_class_resample,
)

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class ShadowModelsManager:
    """Class that creates n ShadowModels, trains them and maintains them."""

    def __init__(
            self, n_models: int, results_path: str, test_size: float, random_state: int,
            model_creator_fn: Callable = create_random_forest, **kwargs
    ) -> None:
        """Creates a new Shadow Models Manager.

        Args:
            n_models (int): Number of models to create.
            results_path (str): Path where to save data.
            test_size (float): Size of the test for the splitting.
            random_state (int): Seed of random number generators.
            model_creator_fn (Callable): function that returns a fitted model on x,y.
        Keyword Args:
            categorical_mask
        """

        self.__n_models = n_models
        self.results_path = results_path
        self.test_size = test_size
        # Creates the results path if it does not exists
        os.makedirs(results_path, exist_ok=True)
        # logger.debug(f"RESULTS PATH CREATED: {results_path}")

        categorical_mask = kwargs.get("categorical_mask", None)

        # SMOTE oversampler
        self.oversampler = SMOTENC(categorical_mask, sampling_strategy="minority",
                                   random_state=random_state) if (categorical_mask is not None) and any(
            categorical_mask) else SMOTE(
            random_state=random_state)
        self.model_creator = model_creator_fn
        self.attack_dataset = None  # set in self.fit

    def get_attack_dataset(self) -> DataFrame:
        """
        Get the dataset for the M.I. Attack Model.

        Notes:
            The attack dataset is initialized at the end of self.fit so this function should not be called before.
        Returns:
            DataFrame containing records of the form (probab. vector, true label, in/out)
        """
        if self.attack_dataset is None:
            raise Exception("The attack dataset is available only after the shadow models have been fit")
        return self.attack_dataset

    def fit(self, x: ndarray, y: ndarray) -> None:
        """Fits a number of shadow models, tests them and initializes self.attack_dataset.


        This method creates n shadow models and saves them and their input data. For each shadow model first it splits
        the data into train and test sets and, after training, inserts in self.attack_dataset tuples of the form
        (prob, y, in) for the data in the train set and (prob, y, out) for the data in the test set.

        Args:
            x (ndarray): Input examples.
            y (ndarray): Target values.
        """
        # List of attack datasets to be concatenated at the end of this loop
        attack_datasets: List[DataFrame] = []
        for i in range(self.__n_models):
            # Train-test splitting
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=self.test_size, stratify=y, train_size=0.6
            )
            # Oversampling of the minority classes

            # smote requires a minimum of samples for each class
            # min_freq = min(frequencies(y_train), key=lambda el: el[1])
            # while min_freq[1] < 10:
            #    x_train, y_train = minority_class_resample(x_train, y_train, 11)
            #    # frequencies of the labels
            #    min_freq = min(frequencies(y_train), key=lambda el: el[1])

            # SMOTE oversampling
            x_train, y_train = self.oversampler.fit_resample(x_train, y_train)

            # Random Forest obtained via grid search
            # TODO split classifier creation from grid search (also change names since now it can also be adaboost)
            rf: RandomForestClassifier = self.model_creator(x_train, y_train)

            # Predictions of the shadow model on the train and test set

            y_pred_train: ndarray = rf.predict(x_train)
            y_prob_train: ndarray = rf.predict_proba(x_train)

            y_pred_test: ndarray = rf.predict(x_test)
            y_prob_test: ndarray = rf.predict_proba(x_test)

            # Classification reports
            report_train: str = classification_report(y_train, y_pred_train)
            report_test: str = classification_report(y_test, y_pred_test)

            # TODO this part should not be in the fit method of the SM, but should be done by the caller
            # Path of the shadow model
            path: str = f"{self.results_path}/{i}"
            # Creates the directory if it does not exists
            os.makedirs(path, exist_ok=True)
            # Saves the reports
            save_txt(f"{path}/report_train.txt", report_train)
            save_txt(f"{path}/report_test.txt", report_test)
            # Creates the dataset for the attack model (prob, label, in/out)
            attack_dataset_i: DataFrame = create_attack_dataset(
                y_prob_train, y_train, y_prob_test, y_test
            )
            # Appends the local attack dataset to the global dataset
            attack_datasets.append(attack_dataset_i)
            # Saves input data
            savez_compressed(
                f"{path}/data",
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
            # Saves the shadow model
            save_pickle_bz2(f"{path}/model.pkl.bz2", rf)

        # Concatenates the attack datasets
        self.attack_dataset: DataFrame = concat(attack_datasets)
