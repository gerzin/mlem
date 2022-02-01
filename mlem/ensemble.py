from typing import List, Sequence
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
