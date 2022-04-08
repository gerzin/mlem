"""
This module contains PyTorch Dataset for the geolocation30 dataset.

This class should be used to load the geotarget_30.csv dataset (or its train and test splits) in order to use it with
PyTorch.
"""
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class GermanCredit(Dataset):

    def __init__(self, csv_file, features_transform=None, target_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with the german data.
            features_transform (callable, optional): Optional transform to be applied
                to the rows containing the features.
            target_transform (callable, optional): Optional transform to be applied
                on a target.
        Notes:
            * the order of the returned elements is (features, target)
            * features is a numpy array
        Examples:
            >>> dataset = GermanCredit("../data/german/german_credit_data.csv")
            >>> for i in range(len(dataset)):
            >>>    feat, targ = dataset[i]
            >>>    ...

        """
        # remove entries with missing values
        self.german = pd.read_csv(csv_file).dropna()
        self.features_transform = features_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.geolocation)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # get the right columns
        features = self.geolocation.iloc[index, 2:-1].to_numpy(dtype=np.float32)
        # get the target variable
        targ = self.geolocation.iloc[index, -1]

        if self.features_transform:
            features = self.features_transform(features)

        if self.target_transform:
            targ = self.target_transform(targ)

        return features, targ
