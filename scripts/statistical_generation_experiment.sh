#!/usr/bin/env bash
cd ..
echo "Going to: $PWD"

echo "Sourcing virtual env"
source venv/bin/activate

time ./adult_randfor_clustered_adaboost.py rf notebooks/datasets/adult/BB_NO_CLUSTERING/BB_DATA/adult_rf_noclustering.bz2 notebooks/datasets/adult/BB_NO_CLUSTERING/BB_DATA/adult_rf_noclustering_data_nsamples_2.npz --results-path "ADULT_STATISTICAL_NO_OVERSAMPLING_AFTER_GEN_2_NO_ATK_OVERS" --statistical-generation
time ./diva_randfor_clustered_adaboost.py rf notebooks/datasets/diva/BB_NO_CLUSTERING/BB_DATA/diva_rf_noclustering.bz2 notebooks/datasets/diva/BB_NO_CLUSTERING/BB_DATA/diva_rf_noclustering_data_nsamples_5.npz --statistical-generation --results-path "DIVA_STATISTICAL_NO_OVERSAMPLING_AFTER_GEN_2_NO_ATK_OVERS"

