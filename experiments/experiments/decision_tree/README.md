# Decision Tree

This folder contains the experiments with a `Decision Tree` used in place of the `LIME`s `Ridge Regressor`.

Folders structure:

* `noisy_dataset` contains the experiments where the noisy "validation" dataset is used for the creation of the shadow models.

* `noisy_dataset_only_closest` contains the experiments where the noisy "validation" dataset is used for the creation of the shadow models, but for each instance to explain, only the points of the noisy dataset closest to the instance (mean distance + 3std) are considered.

* `decision_tree_dataset` contains the experiments where the same dataset used to build the `Decision Tree` is used for the creation of the shadow models.

* `decision_tree_dataset_no3std` contains the experiments where the same dataset used to build the `Decision Tree` is used for the creation of the shadow models but the generation isn't filtered.

* `comparisons` contains the comparison between the results generated in the above folders.