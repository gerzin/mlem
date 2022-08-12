from typing import List, Sequence

import numpy as np
from numpy import ndarray
from numpy.core.fromnumeric import argmax
from numpy.core.shape_base import vstack
from sklearn.base import RegressorMixin


class EnsembleClassifier:
    """Ensemble of regressors managed as a unique classifier."""

    def __init__(self, regressors: Sequence[RegressorMixin]):
        """Creates a new ensemble classifier based on

        Args:
            regressors (Sequence[RegressorMixin]): Regressors that output the probability of
            a certain item to be in a certain class.
        """
        self.regressors = regressors

    def predict(self, x: ndarray) -> ndarray:
        """Predicts a new item based on new data.

        Args:
            x (ndarray): Input examples.

        Returns:
            ndarray: Target values.
        """
        # List of predictions for each regressor
        predictions: List[ndarray] = [reg.predict(x) for reg in self.regressors]
        # Vertically stacks the predictions for each regressor
        y_prob: ndarray = vstack(predictions)
        # Gets the class with maximum probability
        y_pred: ndarray = argmax(y_prob, axis=0)
        return y_pred


class HardVotingClassifier:
    """
        Hard voting classifier for binary data with classes "in" and "out".
    """

    def __init__(self, classifiers: List):
        """

        Args:
            classifiers:
        """
        self.classifiers_ = classifiers
        self.nclass = len(classifiers)
        assert all([x.classes_ == self.classifiers_[0].classes_] for x in self.classifiers_)

    def __eq__(self, other):
        nc = self.nclass == other.nclass
        same_length = len(self.classifiers_) == len(other.classifiers_)
        return nc and same_length and all([x == y for (x, y) in zip(self.classifiers_, other.classifiers_)])

    def predict(self, x):
        """
            Use the classification of the classifiers to classify the elements. The output label is
            the most common label or "even" if the number of classifiers is even and the labels predicted
            are 50% "in" and 50% "out"
        Args:
            x: elements to classify

        Returns:
            array with classification, containing ["in", "out", "even"]
        """

        # use 1 for "in", -1 for "out", sum all the arrays and replace the values => 0 with in
        # the ones < 0 with out, return a boolean mask to indicate the ones where there was a 50/50
        # decision (only for when the number of classifiers is even)
        def convert_to_number(x):
            if x == "in":
                return 1
            elif x == "out":
                return -1
            else:
                raise ValueError(f"invalid argument {x}")

        def convert_to_inout(x):
            if x > 0:
                return "in"
            elif x < 0:
                return "out"
            else:
                return "even"

        convert_to_number_vectorized = np.vectorize(convert_to_number)
        convert_to_inout_vectorized = np.vectorize(convert_to_inout)
        # list of numpy arrays
        predictions = [convert_to_number_vectorized(model.predict(x)) for model in self.classifiers_]

        return convert_to_inout_vectorized(sum(predictions))


class SoftVotingClassifier:
    """
        Soft voting classifier which sums the probabilities predicted by the individual
        classifiers and returns the class with the highest value.
    """

    def __init__(self, classifiers: List):
        """

        Args:
            classifiers:
        """
        self.classifiers_ = classifiers
        self.nclass = len(classifiers)
        self.classes_ = self.classifiers_[0].classes_
        assert all([x.classes_ == self.classifiers_[0].classes_] for x in self.classifiers_)

    def predict(self, x):
        """
        Use the predicted probabilities of each individual classifier to predict the most probable class.

        Args:
            x: elements to classify

        Returns:
            array with the classification
        """

        def extract_class_from_proba(x):
            am = np.argmax(x)
            return self.classes_[am]

        predictions_prob = [model.predict_proba(x) for model in self.classifiers_]
        predictions_prob_sum = sum(predictions_prob)
        return np.apply_along_axis(extract_class_from_proba, 1, predictions_prob_sum)
