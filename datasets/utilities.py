"""
This module contains utilities for handling the datasets.
"""
from sklearn.model_selection import train_test_split

# Seed used by the functions in this module
DEFAULT_SPLITTING_SEED = 1234

#
# INFOS FOR THE GEOLOCATION30 DATASET SPLITS
#

# Split for the Geolocation Dataset
GEOLOCATION_30_TRAIN = 0.8
GEOLOCATION_30_SHUFFLE = True


