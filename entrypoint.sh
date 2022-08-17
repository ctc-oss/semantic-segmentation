#!/usr/bin/env bash

source activate gdal_python36

function download_data {
    python3 -u data/download_datasets.py
}

function check_data_type {
    opt=$1
    if [[ "$opt" == "--s3_only" || $DATA_TYPE == "s3" ]]; then
        python3 -u -c "from utils.data import setup_data; setup_data(False, True);"
        return
    elif [[ "$opt" == "--local_only" || $DATA_TYPE == "local" ]]; then
        python3 -u -c "from utils.data import setup_data; setup_data(True, False);"
        return
    elif [[ "$opt" == "--all" || $DATA_TYPE == "all" ]]; then
        python3 -u -c "from utils.data import setup_data; setup_data(True, True);"
        return
    fi
}

function setup_data {
    if [[ $# < 1 ]]; then
        check_data_type ""
        return
    fi

    for opt in $@; do
        if [[ "$opt" == "--s3_only" || $DATA_TYPE == "s3" ]]; then
            python3 -u -c "from utils.data import setup_data; setup_data(False, True);"
            return
        elif [[ "$opt" == "--local_only" || $DATA_TYPE == "local" ]]; then
            python3 -u -c "from utils.data import setup_data; setup_data(True, False);"
            return
        elif [[ "$opt" == "--all" || $DATA_TYPE == "all" ]]; then
            python3 -u -c "from utils.data import setup_data; setup_data(True, True);"
            return
        fi
    done
}

function train {
    python3 -u train.py
}

function deploy {
    python3 -u deploy.py undeploy
}

function test {
    python3 -u test.py
}

$@
