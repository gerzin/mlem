#!/usr/bin/env bash
cd ..
echo "Going to: $PWD"

echo "Sourcing virtual env"
source venv/bin/activate
# LIME GENERATED
time ./adult_randfor_clustered_adaboost.py rf notebooks/datasets/adult/BB_NO_CLUSTERING/BB_DATA/adult_rf_noclustering.bz2 notebooks/datasets/adult/BB_NO_CLUSTERING/BB_DATA/adult_rf_noclustering_data_nsamples_2.npz --results-path "ADULT_GENERATED_OVERS_AND_ATTACK_OVERS_NEW_FIX_SMOTE" --neighborhood-sampling gaussian
time ./diva_randfor_clustered_adaboost.py rf notebooks/datasets/diva/BB_NO_CLUSTERING/BB_DATA/diva_rf_noclustering.bz2 notebooks/datasets/diva/BB_NO_CLUSTERING/BB_DATA/diva_rf_noclustering_data_nsamples_5.npz --results-path "DIVA__GENERATED_OVERS_AND_ATTACK_OVERS_NEW_FIX_SMOTE" --neighborhood-sampling gaussian

# CENTROIDS
time ./adult_randfor_clustered_adaboost.py rf notebooks/datasets/adult/BB_NO_CLUSTERING/BB_DATA/adult_rf_noclustering.bz2 notebooks/datasets/adult/BB_NO_CLUSTERING/BB_DATA/adult_rf_noclustering_data_nsamples_2.npz --results-path "ADULT_CENTROIDS_LIME_WITH_OVERSAMPLING_AFTER_PICKING_AND_ATTACK_OVERS_NEW_FIX_SMOTE" --local-attack-dataset-path notebooks/datasets/adult/BB_NO_CLUSTERING/BB_DATA/validation_noisy.csv
time ./diva_randfor_clustered_adaboost.py rf notebooks/datasets/diva/BB_NO_CLUSTERING/BB_DATA/diva_rf_noclustering.bz2 notebooks/datasets/diva/BB_NO_CLUSTERING/BB_DATA/diva_rf_noclustering_data_nsamples_5.npz --results-path "DIVA_CENTROIDS_LIME_WITH_OVERSAMPLING_AFTER_PICKING_AND_ATTACK_OVERS_NEW_FIX_SMOTE" --local-attack-dataset-path notebooks/datasets/diva/BB_NO_CLUSTERING/BB_DATA/validation_noisy.csv
# STATISTICAL
time ./adult_randfor_clustered_adaboost.py rf notebooks/datasets/adult/BB_NO_CLUSTERING/BB_DATA/adult_rf_noclustering.bz2 notebooks/datasets/adult/BB_NO_CLUSTERING/BB_DATA/adult_rf_noclustering_data_nsamples_2.npz --results-path "ADULT_STATISTICAL_OVERSAMPLING_AFTER_GEN_AND_ATTACK_OVERS_NEW_FIX_SMOTE" --statistical-generation
time ./diva_randfor_clustered_adaboost.py rf notebooks/datasets/diva/BB_NO_CLUSTERING/BB_DATA/diva_rf_noclustering.bz2 notebooks/datasets/diva/BB_NO_CLUSTERING/BB_DATA/diva_rf_noclustering_data_nsamples_5.npz  --results-path "DIVA_STATISTICAL_OVERSAMPLING_AFTER_GEN_AND_ATTACK_OVERS_NEW_FIX_SMOTE" --statistical-generation

