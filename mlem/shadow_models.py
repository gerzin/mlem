import os
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from numpy import concatenate, ndarray, savez_compressed
from numpy.core.fromnumeric import argmin
from numpy.core.shape_base import vstack
from numpy.lib.arraysetops import unique
from pandas import DataFrame, concat
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler

from mlem.utilities import (
    create_attack_dataset,
    create_random_forest,
    save_pickle,
    save_txt, frequencies,
)

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class ShadowModel(ABC):

    @abstractmethod
    def fit(self, x: ndarray, y: ndarray) -> None:
        """
        Fit the model on some data
        Args:
            x: features
            y: targets

        Returns:
            None
        """
        pass

    @abstractmethod
    def predict(self, x: ndarray) -> ndarray:
        """Predicts an output from an unseen example.

        Args:
            x (ndarray): Input example

        Returns:
            ndarray: Predicted value
        """
        pass

    @abstractmethod
    def predict_proba(self, x: ndarray) -> ndarray:
        """Prediction probability from an unseen example.

        Args:
            x (ndarray): Input example.

        Returns:
            ndarray: Predicted value.
        """
        pass


# TODO this class is using random forests as shadow model.
#      Should probably pass the model to use as argument or at least as option.
class ShadowModelsManager:
    """Class that creates n ShadowModels, trains them and maintains them."""

    def __init__(
            self, n_models: int, results_path: str, test_size: float, random_state: int
    ) -> None:
        """Creates a new Shadow Models Manager.

        Args:
            n_models (int): Number of models to create.
            results_path (str): Path where to save data.
            test_size (float): Size of the test for the splitting.
            random_state (int): Seed of random number generators.
        """

        self.__n_models = n_models
        self.results_path = results_path
        self.test_size = test_size
        # Creates the results path if it does not exists
        os.makedirs(results_path, exist_ok=True)
        logger.debug(f"RESULTS PATH CREATED: {results_path}")

        # SMOTE oversampler
        self.oversampler = SMOTE(random_state=random_state)

        self.attack_dataset = None  # set in self.fit

    # TODO this method isn't calling self. Should probably move it under utilities
    #      also, consider using numba to improve speed
    #      Question: why are we doing this? Wouldn't smote suffice?
    def __minority_class_resample(self, x: ndarray, y: ndarray, n_samples: int = 50) -> Tuple[ndarray, ndarray]:
        """Resamples x and y generating n_samples elements of the minority class.

        Args:
            x (ndarray): Input data.
            y (ndarray): Target labels.
            n_samples (int, optional): Number of samples to be drawn for the minority class. Defaults to 50.

        Returns:
            Tuple[ndarray, ndarray]: Resampled x and y.
        """
        # Unique classes and their frequencies
        classes, occurrences = unique(y, return_counts=True)
        # Minority class
        min_class: int = classes[argmin(occurrences)]
        # Records of every other class
        x_maj, y_maj = x[y != min_class], y[y != min_class]
        # Records of the minority class
        x_min, y_min = x[y == min_class], y[y == min_class]
        # Resamples the minority class
        x_min, y_min = resample(x_min, y_min, n_samples=n_samples)
        # Reconstructs the original dataset
        x = vstack((x_maj, x_min))
        y = concatenate((y_maj, y_min))
        return x, y

    def get_attack_dataset(self) -> DataFrame:
        """
        Get the dataset for the M.I. Attack Model.

        Notes:
            The attack dataset is initialized at the end of self.fit so this function should not be called before.
        Returns:
            DataFrame containing records of the form (probab. vector, true label, in/out)
        """
        return self.attack_dataset

    def fit(self, x: ndarray, y: ndarray) -> None:
        """Fits a number of shadow models, tests them and initializes self.attack_dataset.

        This method creates n shadow models and saves them and their input data

        Args:
            x (ndarray): Input examples.
            y (ndarray): Target values.
        """
        # List of attack datasets to be concatenated at the end of this loop
        attack_datasets: List[DataFrame] = []
        logger.info(f"Fitting {self.__n_models} Shadow Models")
        for i in range(self.__n_models):
            # Train-test splitting
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=self.test_size
            )
            # Oversampling of the minority class

            # smote requires a minimum of samples for each class
            min_freq = min(frequencies(y_train), key=lambda el: el[1])
            while min_freq[1] < 10:
                x_train, y_train = self.__minority_class_resample(x_train, y_train, 11)
                # frequencies of the labels
                min_freq = min(frequencies(y_train), key=lambda el: el[1])

            # SMOTE oversampling
            x_train, y_train = self.oversampler.fit_resample(x_train, y_train)

            logger.debug(f"Creating Random Forest {i}")
            # Random Forest obtained via grid search
            rf: RandomForestClassifier = create_random_forest(x_train, y_train)
            logger.debug(f"Random Forest {i} created")
            # Prediction of the shadow model

            # TODO performance optimization: first predict proba and then assign argmax or something else to y_pred_train and test if using other models and not the RF

            y_pred_train: ndarray = rf.predict(x_train)
            y_prob_train: ndarray = rf.predict_proba(x_train)

            y_pred_test: ndarray = rf.predict(x_test)
            y_prob_test: ndarray = rf.predict_proba(x_test)

            # Classification reports
            report_train: str = classification_report(y_train, y_pred_train)
            report_test: str = classification_report(y_test, y_pred_test)
            logger.debug(f"Predictions and classification report {i} done")

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
            save_pickle(f"{path}/model.pkl.bz2", rf)
            logger.debug(f"Saved {i} Shadow Model")
        # Concatenates the attack datasets
        self.attack_dataset: DataFrame = concat(attack_datasets)
