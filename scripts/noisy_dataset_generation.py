#!/usr/bin/env python3
import pandas as pd
import pickle
import numpy as np
import argparse

HOMEDIR = "../Location30/"
mode = "Location"
class_name = "profile"  # Indica la variabile target?
perc = 5
filename: str = f"{HOMEDIR}data/{mode}_shadow.csv"
dataset_shadow = pd.read_csv(filename, index_col=0)
dataset_shadow.pop(class_name)
dataset_shadow.pop('UserID')

n_rows = dataset_shadow.shape[0]
n_col = dataset_shadow.shape[1]
new_df = pd.DataFrame()

percentage = int((perc / float(100)) * n_rows)  # Posso portarlo fuori dal loop?
for c in range(n_col):
    index_to_replace = np.random.choice(dataset_shadow.index, size=percentage)
    new_values = np.random.rand(percentage)
    for i in range(0, len(index_to_replace)):
        dataset_shadow.iloc[i, c] = new_values[i]

filename: str = f"{HOMEDIR}data/{mode}_noise_shadow.csv"
# dataset_shadow.to_csv(filename)
f = open(filename, 'wb')
pickle.dump(dataset_shadow, f)
f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Insert noise inside a dataset.')
    parser.add_argument('dataset', metavar='DATASET', type=str, nargs=1, required=True,
                        help='an integer for the accumulator')
