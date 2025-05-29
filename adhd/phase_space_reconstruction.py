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
plain_text_folder = os.path.join(dataset_base_path, "plain_text_segments")

assert os.path.exists(plain_text_folder)

converted_sentences_path = os.path.join(dataset_base_path, "sentence_converted")
os.makedirs(converted_sentences_path, exist_ok=True)

L = os.listdir(dataset_base_path)


for l in L:
    path_current_l = os.path.join(dataset_base_path, l)
    if not os.path.isdir(path_current_l):
        continue

    files = os.listdir(path_current_l)
    for file in files:
        if not file.endswith('.npz'):
            continue

        full_path = os.path.join(path_current_l, file)
        npz_data = np.load(full_path, allow_pickle=True)
        microstate_sequence = npz_data['microstate_sequence']

        # Create subfolder if not exist
        subfolder_path = os.path.join(recurrent_sentences_raw_path, l)
        os.makedirs(subfolder_path, exist_ok=True)

        # Generate segments
        segment_generator = FiniteTimeDelaySegmentGenerator(
            data=to_segment_sequence(microstate_sequence, True),
            time_delay=delay,
            n_states=n_states,
            cut=[2, 4096],
            data_with_repetition=True
        )
        segments, repetition = segment_generator.calculate_recurrent_segments()

        np.save(os.path.join(subfolder_path, f'{file.replace(".npz", "")}_seg.npy'),
                np.array(segments, dtype='object'), allow_pickle=True)
        np.save(os.path.join(subfolder_path, f'{file.replace(".npz", "")}_repetitions.npy'),
                np.array(repetition, dtype='object'), allow_pickle=True)
