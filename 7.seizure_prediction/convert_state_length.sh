#!/bin/bash

# Define the base path
BASE_PATH="../data/test/recurrence_sentences/epileptic_eeg_dataset"

# Run the sentence_plain_text_encoder for the epileptic EEG dataset
python ../4.pcfg_training/convert_state_length.py --file_path "$BASE_PATH/seizure_integrated_all_d2_s4_repetition.npy" --output_file_path "$BASE_PATH/seizure_integrated_all_d2_s4_repetition.txt"

# Run the sentence_plain_text_encoder for the normal EEG dataset
python ../4.pcfg_training/convert_state_length.py --file_path "$BASE_PATH/normal_integrated_all_d2_s4_repetition.npy" --output_file_path "$BASE_PATH/normal_integrated_all_d2_s4_repetition.txt"

# Run the sentence_plain_text_encoder for the pre-epileptic EEG dataset
python ../4.pcfg_training/convert_state_length.py --file_path "$BASE_PATH/pre-epileptic_integrated_all_d2_s4_repetition.npy" --output_file_path "$BASE_PATH/pre-epileptic_integrated_all_d2_s4_repetition.txt"

