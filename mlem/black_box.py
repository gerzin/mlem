from abc import ABC, abstractmethod
from numpy import array, ndarray
from torch import tensor, float32, no_grad, argmax
from torch.nn import Module
from sklearn.base import ClassifierMixin


class BlackBox(ABC):
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


class PyTorchBlackBox(BlackBox):
    """Wrapper for a PyTorch black box classifier."""

    def __init__(self, model: Module) -> None:
        self.model = model

    def predict(self, x: ndarray) -> ndarray:
        self.model.eval()
        X_tensor = tensor(x, dtype=float32)
        with no_grad():
            output = self.model(X_tensor)
            y_pred = argmax(output, dim=1)
        return array(y_pred)

    def predict_proba(self, x: ndarray) -> ndarray:
        X_tensor = tensor(x, dtype=float32)
        with no_grad():
            y_prob = self.model(X_tensor)
        return array(y_prob)


class SklearnBlackBox(BlackBox):
    """Wrapper for a scikit-learn based black box classifier."""

    def __init__(self, model: ClassifierMixin) -> None:
        self.model = model

    def predict(self, x: ndarray) -> ndarray:
        return self.model.predict(x)

    def predict_proba(self, x: ndarray) -> ndarray:
        return self.model.predict_proba(x)
