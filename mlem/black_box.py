"""
This module contains wrappers for the PyTorch and SciKit-learn models, used to expose a common interface.
"""
from abc import ABC, abstractmethod
from numpy import array, ndarray
from torch import tensor, float32, no_grad, argmax
from torch.nn import Module
from sklearn.base import ClassifierMixin
import numpy as np
import torch


class BlackBox(ABC):
    """
    Abstract class representing the interface of a Black Box.
    """

    @abstractmethod
    def predict(self, x: ndarray) -> ndarray:
        """Predicts an output from an unseen example.

        Args:
            x (ndarray): Input example.

        Returns:
            ndarray: Predicted value.
        """
        pass

    @abstractmethod
    def predict_proba(self, x: ndarray) -> ndarray:
        """Return the prediction probability vector from an unseen example.

        Args:
            x (ndarray): Input example.

        Returns:
            ndarray: Probability vector.
        """
        pass


class PyTorchBlackBox(BlackBox):
    """Wrapper for a PyTorch black box classifier."""

    def __init__(self, model: Module, activation=None) -> None:
        """
        Args:
            model: model to wrap.
            activation: Activation function to call. If it's integrated into the model use none.
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.activation = activation

    def predict(self, x: ndarray) -> ndarray:
        """
        Uses the wrapped model to compute predictions.

        The tensor built using the argument is automatically moved on the same device the model is on.
        Args:
            x: values to predict.

        Returns:
            array containing the predictions.
        """
        self.model.eval()
        X_tensor = tensor(x, dtype=float32, device=self.device)
        with no_grad():
            output = self.model(X_tensor)
            if self.activation:
                output = self.activation(output)
            y_pred = argmax(output, dim=1)
        return array(y_pred.cpu())

    def predict_proba(self, x: ndarray) -> ndarray:
        """
        Uses the wrapped model to compute prediction probability vectors.
        Args:
            x: values for which to compute the probability vectors.

        Returns:
            array of probability vectors.
        """
        self.model.eval()
        X_tensor = tensor(x, dtype=float32, device=self.device)
        with no_grad():
            if self.activation:
                y_prob = self.activation(self.model(X_tensor))
            else:
                y_prob = self.model(X_tensor)
        return array(y_prob.cpu())


class SklearnBlackBox(BlackBox):
    """Wrapper for a scikit-learn based black box classifier."""

    def __init__(self, model: ClassifierMixin) -> None:
        self.model = model

    def predict(self, x: ndarray) -> ndarray:
        return self.model.predict(x)

    def predict_proba(self, x: ndarray) -> ndarray:
        return self.model.predict_proba(x)


class KerasBlackBoxBin:
    """Wrapper for a keras black box for binary classification."""

    def __init__(self, model) -> None:
        self.model = model

    def predict(self, x):
        return self.model.predict(x).round().ravel()

    def predict_proba(self, x):
        p = self.model.predict(x)
        return np.column_stack((1 - p, p))
