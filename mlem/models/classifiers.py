from abc import ABC, abstractmethod
import numpy as np
from typing import Any
from sklearn.ensemble import RandomForestClassifier
import bz2
import pickle


class MLEMAbstractClassifier:
    _inner_model = None

    @property
    @abstractmethod
    def model(self):
        """
        Get the inner classifier.

        Returns:
            object: the wrapped classifier
        """
        return self._inner_model

    @model.setter
    @abstractmethod
    def model(self, model):
        """
        Set a new object as classifier.

        Args:
            model: classifier
        """
        self._inner_model = model

    @abstractmethod
    def fit(self, x, y, *args, **kwargs):
        """
        Fit a classifier on some data.

        Args:
            x: features
            y: targets
            *args:
            **kwargs:

        Returns:

        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray, *args, **kwargs) -> Any:
        """
        Predict the class of an instance.

        Args:
            x: instance whose class we want to predict.

        Returns:

        """
        pass

    @abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return the prediction probability vector from an unseen example.

        Args:
            x (ndarray): Input example.

        Returns:
            ndarray: Probability vector.
        """
        pass

    @abstractmethod
    def save(self, path, *args, **kwargs):
        """
        Save the model.

        Args:
            path:
            *args:
            **kwargs:

        Returns:

        """
        with bz2.open(path, "wb") as f:
            pickle.dump(self._inner_model, f)

    @classmethod
    @abstractmethod
    def load(cls, path, *args, **kwargs):
        """
        Load the model.


        Args:
            path:
            *args:
            **kwargs:

        Returns:

        """
        print(f"{cls=}")
        data = bz2.BZ2File(path, "rb")
        inner_model = pickle.load(data)
        loaded = cls()
        loaded.model = inner_model
        return loaded


class MLEMRandomForestClassifier(MLEMAbstractClassifier):

    def __init__(self, *args, **kwargs):
        """
        Create a new random forest classifier based on skit-learn's RandomForestClassifier.

        Notes:
            args and kwargs are forwarded to RandomForestClassifier

        Args:
            *args:
            **kwargs:
        """
        self._inner_model = RandomForestClassifier(*args, **kwargs)

    @property
    def model(self):
        return super(MLEMRandomForestClassifier, self).model

    @model.setter
    def model(self, model):
        self._inner_model = model

    def fit(self, x, y, *args, **kwargs):
        self._inner_model.fit(x, y)

    def predict(self, x: np.ndarray, *args, **kwargs) -> Any:
        return self._inner_model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._inner_model.predict_proba(x)

    def save(self, path, *args, **kwargs):
        super(MLEMRandomForestClassifier, self).save(path, *args, **kwargs)

    @classmethod
    def load(cls, path, *args, **kwargs):
        # data = bz2.BZ2File(path, "rb")
        # inner_model = pickle.load(data)
        # loaded = MLEMRandomForestClassifier()
        # loaded.model = inner_model
        # return loaded
        return super(MLEMRandomForestClassifier, cls).load(path)
