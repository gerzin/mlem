"""

"""
from mlem.utilities import load_pickle_bz2
from pathlib import Path


class LocalExplainer:
    pass


class LoreDTLoader:
    def __init__(self, path):
        self.path_ = Path(path)
        assert self.path_.exists()

    def load(self, index: int):
        index = int(index)
        dt_path = self.path_ / f"dt{index}.bz2"
        assert dt_path.exists()
        return LoreDT(path=dt_path)


class LoreDT:
    """
    Wrapper around the decision tree created by LORE
    """

    def __init__(self, path):
        """

        Args:
            path: Path of the decision tree
        """
        self.dt = load_pickle_bz2(path)

    def predict(self, x):
        return self.dt.predict(x)

    def predict_proba(self, x):
        return self.dt.predict_proba(x)
