#!/bin/bash


BINARY_PATH="../lib/pcfg-cky-inside-outside/bin/phase_convert"
CONFIGURATION_BASE_PATH="./phase_space_reconstruction_configs-exp4"
$BINARY_PATH $CONFIGURATION_BASE_PATH/config_sentence_encoding_normal.yaml
$BINARY_PATH $CONFIGURATION_BASE_PATH/config_sentence_encoding_preseizure.yaml
$BINARY_PATH $CONFIGURATION_BASE_PATH/config_sentence_encoding_seizure.yaml
