import numpy as np
import os
import sys
import argparse
import subprocess
from functools import reduce

# Append paths for imports
sys.path.append("../")
sys.path.append("../lib/microstate_lib/code")

from lib.dataset.dataset import *
from lib.dataset.experiment_utils import to_segment_sequence

def get_filename(path):
    return os.path.basename(path)

# Argument parsing
parser = argparse.ArgumentParser(description='Generate microstate samples from ADHD dataset')
parser.add_argument("--dataset_base_path", default="./data/adhd_microstate_dataset_samples", type=str)
args = parser.parse_args()

dataset_base_path = args.dataset_base_path
converted_sentences_path = os.path.join(dataset_base_path, "sentence_converted")

assert os.path.exists(converted_sentences_path)
training_configurations_path = os.path.join(dataset_base_path, "training_configurations")

os.makedirs(training_configurations_path, exist_ok=True)

L = os.listdir(dataset_base_path)
grammar_path = '../data/pcfg/grammar.pcfg'
trained_grammars_save_path = os.path.join(dataset_base_path, "trained_grammars")

for l in L:
    path_current_l = os.path.join(dataset_base_path, l)
    if not os.path.isdir(path_current_l):
        continue

    files = os.listdir(path_current_l)
    for file in files:
        if not file.endswith('.npz'):
            continue

        # Define subfolder path correctly
        subfolder_path = os.path.join(training_configurations_path, l)
        os.makedirs(subfolder_path, exist_ok=True)

        # Path to output YAML configuration file
        full_out_configuration_path = os.path.join(subfolder_path, file.replace('.npz', ".yaml"))

        # Path to the input text file
        full_converted_plain_text_seg_file = os.path.join(converted_sentences_path, l, file.replace('.npz', "_seg_plain_text.txt"))
        os.makedirs(os.path.join(trained_grammars_save_path, l), exist_ok=True)

        # Create the configuration content
        tmp_configuration = f'''\
main:
    grammar_file: "{grammar_path}"
    input:  "{full_converted_plain_text_seg_file}"
    log_intervals: -1
    log_path: "{os.path.join(trained_grammars_save_path, l)}"
    log_f:
        enabled: true
        intervals: 10000
    log_warning_in_training: false
    batch_size_for_parameter_update: -1
    split_data:
        enabled: false
        val_dataset_path: "../../data/recurrence_sentences/epileptic_eeg_dataset/validate_sentences.txt"
        train_dataset_path: "../../data/recurrence_sentences/epileptic_eeg_dataset/train_sentences.txt"
        train_fraction: 0.8
    validation_file: "../data/recurrence_sentences/epileptic_eeg_dataset/sentence_converted/normal_integrated_all_d2_s4_converted_train_validation.txt"
    n_epochs: 10
    limit_n_sentences: -1
    validation_only: false
'''

        with open(full_out_configuration_path, "w") as f:
            f.write(tmp_configuration)
