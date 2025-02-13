#!/usr/bin/env bash
cd ..
echo "Going to: $PWD"

echo "Sourcing virtual env"
source venv/bin/activate
# CENTROIDS
#time ./adult_randfor_clustered_adaboost.py rf notebooks/datasets/adult/BB_NO_CLUSTERING/BB_DATA/adult_rf_noclustering.bz2 notebooks/datasets/adult/BB_NO_CLUSTERING/BB_DATA/adult_rf_noclustering_data_nsamples_2.npz --results-path "DEBUG_ADULT_NO_OVERS" --local-attack-dataset-path notebooks/datasets/adult/BB_NO_CLUSTERING/BB_DATA/validation_noisy.csv --num-shadow-models 2
python diva_debug_dt.py rf scripts/DEBUG_DATA/diva_dt.bz2 . --results-path "DEBUG_DIVA" --local-attack-dataset-path scripts/DEBUG_DATA/diva_stat_original_dt_shadow_labelled.csv --num-shadow-models 2
