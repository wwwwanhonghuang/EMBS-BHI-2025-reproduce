#!/bin/bash

grammar_files=(
"../data/pcfg-exp2/grammar_log_partition_1_epoch_0.pcfg"
"../data/pcfg-exp2/grammar_log_partition_1_epoch_1.pcfg"
"../data/pcfg-exp2/grammar_log_partition_1_epoch_2.pcfg"
"../data/pcfg-exp2/grammar_log_partition_1_epoch_3.pcfg"
"../data/pcfg-exp2/grammar_log_partition_1_epoch_4.pcfg"
"../data/pcfg-exp2/grammar_log_partition_1_epoch_5.pcfg"
"../data/pcfg-exp2/grammar_log_partition_1_epoch_6.pcfg"
"../data/pcfg-exp2/grammar_log_partition_1_epoch_7.pcfg"
"../data/pcfg-exp2/grammar_log_partition_1_epoch_8.pcfg"
"../data/pcfg-exp2/grammar_log_partition_1_epoch_9.pcfg"
)

# Input files that will be used with each grammar file
sentences_file="../data/recurrence_sentences/epileptic_eeg_dataset/sentence_converted/normal-only/normal_integrated_all_d2_s4_converted_train_validation.txt"
repetitions_file="../data/recurrence_sentences/epileptic_eeg_dataset/sentence_converted/normal-only/normal_integrated_all_d2_s4_converted_train_validation_repetition.txt"

# Check existence of input files first
if [ ! -f "$sentences_file" ]; then
    echo "Error: Sentences file not found at $sentences_file" >&2
    exit 1
fi

if [ ! -f "$repetitions_file" ]; then
    echo "Error: Repetitions file not found at $repetitions_file" >&2
    exit 1
fi

# Check existence of all grammar files
missing_files=0
for grammar_file in "${grammar_files[@]}"; do
    if [ ! -f "$grammar_file" ]; then
        echo "Warning: Grammar file not found - $grammar_file" >&2
        ((missing_files++))
    fi
done

if [ $missing_files -gt 0 ]; then
    echo "Warning: $missing_files grammar files are missing" >&2
    read -p "Continue with remaining files? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Main processing loop
for grammar_file in "${grammar_files[@]}"; do
    if [ -f "$grammar_file" ]; then
        echo "Processing $grammar_file..."
        bash generate_config.sh "$grammar_file" "$sentences_file" "$repetitions_file"
	../lib/pcfg-cky-inside-outside/bin/train_pcfg ./config-tmp.yaml
    else
        echo "Skipping missing file: $grammar_file" >&2
    fi
done

echo "Processing complete."
