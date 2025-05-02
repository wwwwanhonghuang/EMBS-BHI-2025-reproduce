BINARY_FILE=../lib/pcfg-cky-inside-outside/bin/train_pcfg
CONFIGURATION_FILE_PATH=./training_configurations/config_train.yaml

mkdir -p data/logs
$BINARY_FILE $CONFIGURATION_FILE_PATH
