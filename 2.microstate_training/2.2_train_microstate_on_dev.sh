#!/bin/bash

FORCE_REPREPROCESSING=0
CONFIGURATION_FILE=./configs/config-all-person-microstate-dev.json

FLAGS="--database_index_configuration $CONFIGURATION_FILE"

if [ "$FORCE_REPREPROCESSING" -eq 1 ]; then
    FLAGS="$FLAGS --force_repreprocessing"
fi

python microstate_extraction.py $FLAGS
