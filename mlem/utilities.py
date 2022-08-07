import bz2
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Iterable, Callable
import numpy as np
from numpy import ndarray, concatenate
from pandas import concat
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.model_selection import RandomizedSearchCV
from numpy.core.fromnumeric import argmin
from numpy.core.shape_base import vstack
from numpy.lib.arraysetops import unique
from sklearn.utils import resample
from sklearn.ensemble import AdaBoostClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import pandas as pd
import scipy.spatial.distance as distance
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTENC, SMOTE


def save_pickle_bz2(path: str, object: Any):
    """Saves a pickle file compressed in BZ2 format.

    Args:
        path (str): Path where to save object.
        object (Any): Object to save.
    """
    with bz2.open(path, "wb") as f:
        pickle.dump(object, f)


def load_pickle_bz2(path):
    """
    Loads data saved with save_pickle_bz2

    Args:
        path (str): Path where the data are located

    Returns:
        loaded object
    """
    with bz2.BZ2File(path) as f:
        data = pickle.load(f)
    return data


def save_txt(path, txt: str):
    """Saves a text file on disk.

    Args:
        path (str | Path): Path where to save the string.
        txt (str): String to save.
    """
    path = str(path)
    with open(path, "w") as f:
        f.write(txt)


def save_train_test_datasets(path: str, x_train: ndarray, y_train: ndarray, x_test: ndarray, y_test: ndarray) -> None:
    """
    Saves on a compressed file the training and test sets with the respective labels.
    Args:
        path:
        x_train: training features.
        y_train: training targets.
        x_test: test features.
        y_test: test targets.

    Returns:
        None
    """
    np.savez_compressed(
        path,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )


def load_train_test_datasets(path: str) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Loads training and test sets, with the respective labels, saved with save_train_test_datasets.
    Args:
        path: path of the file containing the datasets.

    Returns:
        (x_train: ndarray, y_train: ndarray, x_test: ndarray, y_test: ndarray) - training features, training targets, test features, test targets
    """
    loaded = np.load(path, allow_pickle=True)
    x_train: ndarray = loaded["x_train"]
    y_train: ndarray = loaded["y_train"]
    x_test: ndarray = loaded["x_test"]
    y_test: ndarray = loaded["y_test"]

    return x_train, y_train, x_test, y_test


# Hyperparameters of the grid search
__HYPERPARAMETERS = {
    "bootstrap": [True, False],
    "max_depth": [100, 350, 500],
    "min_samples_leaf": [10, 20, 50],
    "min_samples_split": [5, 10, 50],
    "n_estimators": [100, 350, 500],
    "criterion": ["gini", "entropy"],
}


def create_random_forest(
        x_train: ndarray,
        y_train: ndarray,
        hyperparameters: Dict[str, List[Any]] = __HYPERPARAMETERS,
        n_jobs=4,
        use_halving=True
) -> RandomForestClassifier:
    """Creates a random forest classifier via grid search.

    Args:
        x_train (ndarray): Training input examples.
        y_train (ndarray): Training target values.
        hyperparameters (Dict[str, List[Any]], optional): Dictionary of hyperparameters for the grid search. Defaults to the fixed ones.
        n_jobs: Number of jobs to run in parallel in the grid search. (default 4)
        use_halving (bool): If true use the HalvingGridSearch

    Returns:
        RandomForestClassifier: Random forest classifier.
    """

    rf = RandomForestClassifier()

    if use_halving:
        clf = HalvingGridSearchCV(rf, hyperparameters, refit=True, n_jobs=n_jobs, verbose=0)
    else:
        clf = RandomizedSearchCV(rf, hyperparameters, refit=True, n_jobs=n_jobs, verbose=0)
    clf.fit(x_train, y_train)
    # print(f"GRID_SEARCH BEST PARAMS: {clf.best_params_=}")
    return clf.best_estimator_


__ADA_HYPER = {
    "n_estimators": [50, 100, 150, 200]
}


def create_adaboost(x_train: ndarray,
                    y_train: ndarray,
                    hyperparameters: Dict[str, List[Any]] = __ADA_HYPER,
                    n_jobs=4,
                    use_halving=True
                    ) -> RandomForestClassifier:
    """Creates a AdaBoost classifier.

    Args:
        x_train (ndarray): Training input examples.
        y_train (ndarray): Training target values.
        hyperparameters (Dict[str, List[Any]], optional): Dictionary of hyperparameters for the grid search. Defaults to the fixed ones.
        n_jobs: Number of jobs to run in parallel in the grid search. (default 4)
        use_halving (bool): If true use the HalvingGridSearch

    Returns:
        RandomForestClassifier: Random forest classifier.
    """

    ab = AdaBoostClassifier()
    if use_halving:
        clf = HalvingGridSearchCV(ab, hyperparameters, refit=True, n_jobs=n_jobs, verbose=0)
    else:
        clf = RandomizedSearchCV(ab, hyperparameters, refit=True, n_jobs=n_jobs, verbose=0)
    clf.fit(x_train, y_train)
    return clf.best_estimator_


def __create_local_attack_dataset(
        y_prob: ndarray,
        y: ndarray,
        inout: str,
) -> DataFrame:
    """Creates a dataframe of the form (y_prob, y, "in" / "out") to be used for
    the creation of a dataset for the Attack Model of the Membership Inference.



    Args:
        y_prob (ndarray): Prediction probability for the shadow model.
        y (ndarray): Target label predicted by the shadow model.
        inout (str): Whether these data come from the training test ("in") or test set ("out").

    Returns:
        DataFrame: Local attack dataset. The dataframe contains one column for each class with its respective probability,
        plus a column "label" with the real label and a column "inout" with the value "in" or "out"
    """
    df = DataFrame(y_prob)
    df["label"] = y
    df["inout"] = inout
    return df


def create_attack_dataset(
        y_prob_train: ndarray,
        y_train: ndarray,
        y_prob_test: ndarray = None,
        y_test: ndarray = None,
) -> DataFrame:
    """Creates a dataset for the Membership Inference Attack Model.
    Creates an attack dataset by creating tuples of the form (y_prob_train, y_train, "in") for the
    data belonging to the train set of the model we are trying to attack and (y_prob_test, y_test, "out")
    for the data in the test set of that model.
    Args:
        y_prob_train (ndarray): Predicted probabilities on the training set.
        y_train (ndarray): Labels of the training set.
        y_prob_test (ndarray, optional): Predicted probabilities on the test set.
        y_test (ndarray, optional): Labels of the test set.
    Returns:
        DataFrame: Local attack model.
    """
    df_in: DataFrame = __create_local_attack_dataset(y_prob_train, y_train, "in")
    if y_prob_test is not None and y_test is not None:
        df_out: DataFrame = __create_local_attack_dataset(y_prob_test, y_test, "out")
        return concat([df_in, df_out])
    return df_in


def minority_class_resample(x: ndarray, y: ndarray, n_samples: int = 50) -> Tuple[ndarray, ndarray]:
    """Resamples x and y generating n_samples elements of the minority class.

    Args:
        x (ndarray): Input data.
        y (ndarray): Target labels.
        n_samples (int, optional): Number of samples to be drawn for the minority class. Defaults to 50.

    Returns:
        Tuple[ndarray, ndarray]: Resampled x and y.
    """
    # Unique classes and their frequencies
    classes, occurrences = unique(y, return_counts=True)
    # Minority class
    min_class: int = classes[argmin(occurrences)]
    # Records of every other class
    x_maj, y_maj = x[y != min_class], y[y != min_class]
    # Records of the minority class
    x_min, y_min = x[y == min_class], y[y == min_class]
    # Resamples the minority class
    x_min, y_min = resample(x_min, y_min, n_samples=n_samples)
    # Reconstructs the original dataset
    x = vstack((x_maj, x_min))
    y = concatenate((y_maj, y_min))
    return x, y


def frequencies(values: Union[Iterable, int, float]) -> np.array:
    """
    Given a set of values return an array containing the unique values and their respective frequencies.

    >>> a = np.array([1 1 1 2 2 2 3 4 4 5 4 5])
    >>> print(frequencies(a))
    >>> array([[1, 3],
    >>>       [2, 3],
    >>>       [3, 1],
    >>>       [4, 3],
    >>>       [5, 2]])

    Args:
        values: array or list from which to extract the unique values and their frequencies
    Returns:
        array - array of [unique value, its frequency] pairs
    """
    # unique, counts
    u, c = np.unique(values, return_counts=True)
    return np.asarray((u, c)).T


def frequency_based_noise(column, size):
    """
    Sample values from a column with replacement.

    Args:
        column: column to sample from
        size: number of elements to sample

    Returns:
        Array of samples
    """
    return column.sample(size, replace=True).to_numpy()


def insert_noise_categorical(dataset: pd.DataFrame, perc: float = 0.1,
                             noise_generating_function: Callable[[pd.Series, int], np.array] = frequency_based_noise):
    """
    Insert noise in a categorical dataset and returns the dataset passed as argument.

    Args:
        dataset (DataFrame): dataset on which to insert the noise ( it should only contain categorical variables )
        perc (float): percentage of noise in the range [0,1]
       noise_generating_function (Callable[[int], array]): function used to generate the noise, must take as input the number of noisy values to
                                   generate inside an argument named size and return an array containing the random values.

    Returns:
        dataset
    """
    n_rows, n_col = dataset.shape
    percentage = int(perc * n_rows)

    for c in range(n_col):
        index_to_replace = np.random.choice(dataset.index,
                                            size=percentage)
        new_values = noise_generating_function(dataset[dataset.columns[c]], size=percentage)
        assert (len(index_to_replace) == len(new_values))
        for ind, val in zip(index_to_replace, new_values):
            dataset.iloc[ind, c] = val
    return dataset


def insert_noise_numerical(dataset: DataFrame, perc: float = 0.1,
                           noise_generating_function: Callable[[int], np.array] = np.random.normal):
    """
    Insert noise in a numerical dataset and returns the dataset passed as argument.

    Args:
        dataset (DataFrame): dataset on which to insert the noise.
        perc (float): percentage of noise in the range [0,1]
        noise_generating_function (Callable[[int], array]): function used to generate the noise, must take as input the number of noisy values to
                                   generate inside an argument named size and return an array containing the random values.


    Examples:

        >>> df = pd.DataFrame(data={'col1': 10 * [1], 'col2': 10 * [2], 'col3': 10 * [3]})
        >>> df[NUMERICAL] = insert_noise_numerical(df[NUMERICAL].copy(), perc=0.1, noise_generating_function=np.random.rand) # note np.random.rand has a size parameter

    """
    n_rows, n_col = dataset.shape
    percentage = int(perc * n_rows)

    for c in range(n_col):
        index_to_replace = np.random.choice(dataset.index,
                                            size=percentage)
        new_values = noise_generating_function(size=percentage)
        assert (len(index_to_replace) == len(new_values))
        for ind, val in zip(index_to_replace, new_values):
            dataset.iloc[ind, c] = val
    return dataset


def sample_from_quantile(data, centroid, nsamples):
    """
    Samples closest points to centroids based on quantiles.
    Args:
        data: data belonging to the cluster indexed by the centroid.
        centroid: centroid
        nsamples: number of samples per quantile.

    Returns:

    """
    distances = distance.cdist(data, centroid, metric="euclidean")
    df = pd.DataFrame(data)
    df['Distances'] = distances
    labels = ['q1', 'q2', 'q3', 'q4']
    df['Quantiles'] = pd.qcut(df.Distances, q=4, labels=labels)
    out = pd.concat([df[df['Quantiles'].eq(label)].sample(nsamples) for label in labels])
    return out.drop(labels=['Distances', 'Quantiles'], axis=1)


def report_and_confusion(y_true, y_pred, ax=None):
    """
    Print the classification report and plot the confusion matrix
    Args:

        y_true: true labels
        y_pred: predicted labels
        ax: axes to plot on, if None a new one is created

    Returns:

    """
    print(classification_report(y_true, y_pred))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)


def split_probs_array(arr):
    """
    Given an array of probabilities, splits it into sub-arrays based on the most probable class.

    Example:
    an array like this:
        [[0.6, 0.4],
         [0.0, 1.0],
         [0.2, 0.8]]

    will be split into these two arrays:

        1) [[0.6, 0.4]]
        2) [[0.0, 1.0],
            [0.2, 0.8]]
    """
    n_classes = len(arr[0])
    separated = []
    max_index_row = np.argmax(arr, axis=1)
    for c in range(n_classes):
        separated.append(arr[max_index_row == c])
    return separated


def compute_centroids(data: np.ndarray, axis=0):
    """
    Returns the centroid of a matrix.
    Args:
        data: matrix
        axis: axis along which compute the centroid
    Returns:

    """
    return data.mean(axis=axis)


ADULT_COLUMN_MASK = np.array([int(x) == 1 for x in "1,0,1,0,1,0,0,0,0,0,1,1,1,0".split(",")])


def create_attack_dataset_from_lime_centroids(lime_x, lime_y, noisy_set, black_box,
                                              categorical_mask,
                                              target_column_name='Target'):
    """
    Given the x and y generated by lime, it extracts from the dataset called noisy_set the elements closest to the respective centroids.

    Where closest -> <= mean + std distance of the points in the cluster from the respective centroid.

    Args:
        lime_x: lime generated features
        lime_y: lime prediction of the features
        noisy_set:
        black_box: black box used to assign a label to the noisy_set
        categorical_mask: mask indicating which features are categorical.
        target_column_name: name of the target column

    Returns:
        Subset of the noisy_set containing points closest to the centroids.
    """

    def dist_from_centroid(data_np, centroid_np):
        """
        Utility function to compute the distance between the rows of a matrix and a point
        Args:
            data_np:
            centroid_np:

        Returns:

        """
        return distance.cdist(data_np, np.array([centroid_np])).flatten()

    # create DataFrame from lime's data
    df_lime = pd.DataFrame(lime_x)
    assert len(categorical_mask) == lime_x.shape[1]
    df_lime[target_column_name] = lime_y
    # Select only numerical columns
    numerical_columns = [col for (col, cat_mask) in zip(df_lime.columns[:-1], categorical_mask) if cat_mask == False]

    df_lime_numerical = df_lime[numerical_columns + [target_column_name]]

    # Separate dataset by target
    class_0 = df_lime_numerical[df_lime_numerical[target_column_name] == 0]
    class_1 = df_lime_numerical[df_lime_numerical[target_column_name] == 1]

    # Compute Centroids
    centroid_0, centroid_1 = class_0.drop(target_column_name, axis=1).mean(), class_1.drop(target_column_name,
                                                                                           axis=1).mean()

    # Compute mean and std of the distance between the points and the respective centroids
    centroid_0_dist = dist_from_centroid(class_0.drop(labels=[target_column_name], axis=1).to_numpy(),
                                         centroid_0.to_numpy())
    centroid_0_mean, centroid_0_std = centroid_0_dist.mean(), centroid_0_dist.std()

    centroid_1_dist = dist_from_centroid(class_1.drop(labels=[target_column_name], axis=1).to_numpy(),
                                         centroid_1.to_numpy())
    centroid_1_mean, centroid_1_std = centroid_1_dist.mean(), centroid_1_dist.std()

    # Create the thresholds for both classes as mean + std
    threshold_0 = centroid_0_mean + centroid_0_std
    threshold_1 = centroid_1_mean + centroid_1_std

    # Use the bb to get a label
    noisy_df = pd.DataFrame(noisy_set)
    noisy_df[target_column_name] = black_box.predict(noisy_df.to_numpy())
    # Now I separate the noisy_df into class 0 and 1, then I keep only the elements with distance from the resp. centroid <= resp. threshold
    numerical_noisy = noisy_df[numerical_columns + [target_column_name]]

    noisy_0 = numerical_noisy[numerical_noisy[target_column_name] == 0]

    noisy_1 = numerical_noisy[numerical_noisy[target_column_name] == 1]
    # keep the closest elements
    noisy_0_closest = noisy_0[
        dist_from_centroid(noisy_0.drop(target_column_name, axis=1).to_numpy(), centroid_0.to_numpy()) <= threshold_0]
    noisy_1_closest = noisy_1[
        dist_from_centroid(noisy_1.drop(target_column_name, axis=1).to_numpy(), centroid_1.to_numpy()) <= threshold_1]

    elements_idx = noisy_0_closest.index.append(noisy_1_closest.index)
    final_elems = noisy_df.iloc[elements_idx]
    assert len(final_elems) == (len(noisy_0_closest) + len(noisy_1_closest))
    distr = final_elems[target_column_name].value_counts(normalize=True)
    print(
        f"[INFO] Selected closest elements to centroids. Tot elem {len(final_elems)} | 0/1 = {distr[0]:.2f}%/{distr[1]:.2f}%")
    return final_elems.drop(target_column_name, axis=1).to_numpy()


def oversample(x, y, categorical_mask, random_state=123):
    """
    https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
    """
    nelems = len(y)
    assert len(x) == nelems

    print(f"{categorical_mask=}")

    sampling_strategy = "minority"

    uniqcls = np.unique(y)
    # {class: %}
    classes_perc = {}
    for c in uniqcls:
        perc = sum([x == c for x in y]) / len(y)
        classes_perc[c] = perc
    # class of min. percentage
    minority_class_value = min(classes_perc.values())
    minority_classes = [c for c, v in classes_perc.items() if v == minority_class_value]

    oversampler = SMOTENC(categorical_mask, sampling_strategy=sampling_strategy, random_state=random_state) if any(
        categorical_mask) else SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    print(f"{x.shape=}\n{y.shape=}")
    X_new, y_new = oversampler.fit_resample(x, y)

    return X_new, y_new


def print_label_distr(y_tr, y_te, lab):
    lab_tr, count_tr = np.unique(y_tr, return_counts=True)
    lab_te, count_te = np.unique(y_te, return_counts=True)

    freq_tr = count_tr / count_tr.sum()
    freq_te = count_te / count_te.sum()

    assert freq_tr.sum() > 0.999 and freq_te.sum() > 0.999
    sep = "* " * 14
    print(f"[INFO] AttackModel Label {lab}:\n{sep}")
    for l, f in zip(lab_tr, freq_tr):
        print(f"* [TRAIN] lab={l} freq={f:.2} *")
    print(sep)
    for l, f in zip(lab_te, freq_te):
        print(f"* [TEST ] lab={l} freq={f:.2} *")
    print(sep)
