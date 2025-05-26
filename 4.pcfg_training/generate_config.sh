#!/bin/bash

# Check arguments
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <grammar_file_path> <input_file_path> <repetion_file_path>"
  exit 1
fi

GRAMMAR_FILE="$1"
INPUT_FILE="$2"

# Output file name
CONFIG_FILE="config-tmp.yaml"

# Generate YAML configuration
cat <<EOF > "$CONFIG_FILE"
main:
  grammar_file: "$GRAMMAR_FILE"
  input:  "$INPUT_FILE"
  log_intervals: -1
  log_path: "data/logs/"
  log_f:
    enabled: false
    intervals: -1
  log_warning_in_training: false
  batch_size_for_parameter_update: -1
  split_data:
    enabled: false
    val_dataset_path: "../../data/recurrence_sentences/epileptic_eeg_dataset/validate_sentences.txt" 
    train_dataset_path: "../../data/recurrence_sentences/epileptic_eeg_dataset/train_sentences.txt"
    train_fraction: 0.8
  validation_file: "$INPUT_FILE"
  n_epochs: 1
  limit_n_sentences: -1
  validation_only: true
  grammar_weight_origin_in_log_form: true
  print_grammar_when_loading: false
EOF

echo "Configuration written to $CONFIG_FILE"
