#!/bin/bash

MODEL_NAMES=("abalone" "airline" "airline-ohe" "covtype" "epsilon" "higgs" "letters" "year_prediction_msd")
BATCH_SIZES=("512" "1024" "2048" "4096" "8192" "16384")
for model_name in "${MODEL_NAMES[@]}"
do
    for batch_size in "${BATCH_SIZES[@]}"
    do
        echo "Running $model_name for batch size $batch_size"
        echo
        MODEL_FILE_NAME=$model_name"_xgb_model_save.json.txt"
        DATA_FILE_NAME=$MODEL_FILE_NAME".test.sampled.txt"
        ./Tahoe $1/$MODEL_FILE_NAME $1/$batch_size/$DATA_FILE_NAME
    done
done
