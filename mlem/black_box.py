"""
This module contains wrappers for the PyTorch and SciKit-learn models, used to expose a common interface.
"""
from abc import ABC, abstractmethod
from numpy import array, ndarray
from torch import tensor, float32, no_grad, argmax
from torch.nn import Module
from sklearn.base import ClassifierMixin
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

    def __init__(self, model: Module, device: torch.device = None) -> None:
        """
        Args:
            model: model to wrap.
            device: device on which to move the model.
        """
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not device else device
        model.to(self.device)

    def predict(self, x: ndarray) -> ndarray:
        """
        Uses the wrapped model to compute predictions.
        Args:
            x: values to predict.

        Returns:
            array containing the predictions.
        """
        self.model.eval()
        X_tensor = tensor(x, dtype=float32, device=self.device)
        with no_grad():
            output = self.model(X_tensor)
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
        X_tensor = tensor(x, dtype=float32, device=self.device)
        with no_grad():
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
