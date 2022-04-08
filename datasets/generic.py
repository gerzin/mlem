"""
Helper class to load data into pytorch
"""
import torch
import numpy as np
from torch.utils.data import Dataset


class GenericDataset(Dataset):
    """

    """

    def __init__(self, features, target, features_transform=None, target_transform=None):
        assert (len(features) == len(target))
        self.feat = features
        self.targ = target
        self.features_transform = features_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.feat)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        f = self.feat[idx]
        t = self.targ[idx]

        if self.features_transform:
            f = self.features_transform(f)

        if self.target_transform:
            t = self.target_transform(t)

        return f, t
