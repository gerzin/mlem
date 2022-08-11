#!/usr/bin/env bash
cd ..
echo "Going to: $PWD"

echo "Sourcing virtual env"
source venv/bin/activate


time ./adult_randfor_clustered_adaboost.py rf notebooks/datasets/adult/BB_NO_CLUSTERING/BB_DATA/adult_rf_noclustering.bz2 notebooks/datasets/adult/BB_NO_CLUSTERING/BB_DATA/adult_rf_noclustering_data_nsamples_2.npz --results-path "ADULT_CENTROIDS_LIME_WITH_OVERSAMPLING_AFTER_PICKING_AND_ATTACK_OVERS" --local-attack-dataset-path notebooks/datasets/adult/BB_NO_CLUSTERING/BB_DATA/validation_noisy.csv --n-jobs=4
time ./diva_randfor_clustered_adaboost.py rf notebooks/datasets/diva/BB_NO_CLUSTERING/BB_DATA/diva_rf_noclustering.bz2 notebooks/datasets/diva/BB_NO_CLUSTERING/BB_DATA/diva_rf_noclustering_data_nsamples_5.npz --results-path "DIVA_CENTROIDS_LIME_WITH_OVERSAMPLING_AFTER_PICKING_AND_ATTACK_OVERS" --local-attack-dataset-path notebooks/datasets/diva/BB_NO_CLUSTERING/BB_DATA/validation_noisy.csv
time ./adult_randfor_clustered_adaboost.py rf notebooks/datasets/adult/BB_NO_CLUSTERING/BB_DATA/adult_rf_noclustering.bz2 notebooks/datasets/adult/BB_NO_CLUSTERING/BB_DATA/adult_rf_noclustering_data_nsamples_2.npz --results-path "ADULT_STATISTICAL_OVERSAMPLING_AFTER_GEN_AND_ATTACK_OVERS" --statistical-generation --n-jobs=4
time ./diva_randfor_clustered_adaboost.py rf notebooks/datasets/diva/BB_NO_CLUSTERING/BB_DATA/diva_rf_noclustering.bz2 notebooks/datasets/diva/BB_NO_CLUSTERING/BB_DATA/diva_rf_noclustering_data_nsamples_5.npz  --results-path "DIVA_STATISTICAL_OVERSAMPLING_AFTER_GEN_AND_ATTACK_OVERS" --statistical-generation

