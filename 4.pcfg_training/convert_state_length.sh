#!/bin/bash

if [ -z "$BASE_PATH" ]; then
  echo "Error: BASE_PATH is not set."
  exit 1
fi

echo "BASE_PATH=$BASE_PATH"

# Run the sentence_plain_text_encoder for the epileptic EEG dataset
python convert_state_length.py --file_path "$BASE_PATH/seizure_integrated_all_d2_s4_repetition.npy" --output_file_path "$BASE_PATH/sentence_converted/seizure_integrated_all_d2_s4_repetition_converted.txt"

# Run the sentence_plain_text_encoder for the normal EEG dataset
python convert_state_length.py --file_path "$BASE_PATH/normal_integrated_all_d2_s4_repetition.npy" --output_file_path "$BASE_PATH/sentence_converted/normal_integrated_all_d2_s4_repetition_converted.txt"

# Run the sentence_plain_text_encoder for the pre-epileptic EEG dataset
python convert_state_length.py --file_path "$BASE_PATH/pre-epileptic_integrated_all_d2_s4_repetition.npy" --output_file_path "$BASE_PATH/sentence_converted/pre-epileptic_integrated_all_d2_s4_repetition_converted.txt"

