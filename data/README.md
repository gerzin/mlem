# DATA

This directory contains various datasets.

# CONVENTIONS

Each dataset is stored whole in its own folder, saved as a CSV file. Optionally in this folder there are two
subfolders, `train` and `test` in which there are the train set and test set obtained by splitting the whole dataset.

The train set is called `train.csv` and the test set `test.csv`. These are the sets used to train and test the models
in `../blackboxes`.

For each dataset there should (probably) be a PyTorch Dataset class under `../datasets`
in a module with the same name of the dataset to load. The dataset in this folder are saved raw, an eventual
pre-processing.

## Datasets

`german`:

- https://www.kaggle.com/uciml/german-credit?select=german_credit_data.csv

`adult` :

- https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
  this dataset has been encoded using the target encoder. (see notebook BlackBoxesForAdult)

## SPLITS REPRODUCIBILITY

### geotarget, adult

    train_test_split(df, train_size=0.8, shuffle=True, random_state=1234)

