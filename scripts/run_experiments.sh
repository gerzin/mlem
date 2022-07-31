#!/usr/bin/env bash
cd ..
echo "Going to: $PWD"

echo "Sourcing virtual env"
source venv/bin/activate

echo "Running experiments on Adult with Adaboost"
time ./adult_randfor_clustered_adaboost.py rf notebooks/datasets/adult/BB_NO_CLUSTERING/BB_DATA/adult_rf_noclustering.bz2 notebooks/datasets/adult/BB_NO_CLUSTERING/BB_DATA/adult_rf_noclustering_data_nsamples_2.npz --results-path "ADULT_OVERSAMPLING_GENERATED_DATASET" --neighborhood-sampling "gaussian" --n-rows 1
#echo "Running experiments on Diva with Adaboost"
#time ./diva_randfor_clustered_adaboost.py rf notebooks/datasets/diva/BB_NO_CLUSTERING/BB_DATA/diva_rf_noclustering.bz2 notebooks/datasets/diva/BB_NO_CLUSTERING/BB_DATA/diva_rf_noclustering_data_nsamples_5.npz --results-path "DIVA_OVERSAMPLING_GENERATED_DATASET" --neighborhood-sampling "gaussian" #--local-attack-dataset-path notebooks/datasets/diva/BB_NO_CLUSTERING/BB_DATA/validation_noisy.csv