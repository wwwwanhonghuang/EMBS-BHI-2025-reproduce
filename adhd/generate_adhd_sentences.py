import numpy as np
import os
import sys
import argparse
import subprocess
from functools import reduce
from tqdm import tqdm
# Append paths for imports
sys.path.append("../")
sys.path.append("../lib/microstate_lib/code")
sys.path.append("../3.phase_space_reconstruction")

from segmentation_module import (
    InfinitePhaseSpaceReonstructionBasedSegmentGenerator,
    FiniteTimeDelaySegmentGenerator,
    FiniteTimeDelayEEGSegmentGenerator
)
from lib.dataset.dataset import *
from lib.dataset.experiment_utils import to_segment_sequence
from data_utils import match_reorder_topomaps

def get_filename(path):
    return os.path.basename(path)

# Argument parsing
parser = argparse.ArgumentParser(description='Generate microstate samples from ADHD dataset')
parser.add_argument("--dataset_base_path", default="./data/syntax_analysis_data/adhd", type=str)
args = parser.parse_args()

dataset_base_path = args.dataset_base_path

plain_text_folder = os.path.join(dataset_base_path, "..", "plain_text_segments")
os.makedirs(plain_text_folder, exist_ok=True)

recurrent_sentences_raw_path = os.path.join(dataset_base_path, "..", "recurrent_sentences_raw")
os.makedirs(recurrent_sentences_raw_path, exist_ok=True)

L = os.listdir(dataset_base_path)

# === Step 1: Generate segments and repetitions ===
print("=== Step 1: Generate segments and repetitions ===")
delay = 2
n_states = 4

path_current = dataset_base_path

if not os.path.isdir(path_current):
    raise Exception()

files = os.listdir(path_current)
print(path_current, files)
for file in files:
    if not file.endswith('.npz'):
        continue

    full_path = os.path.join(path_current, file)
    npz_data = np.load(full_path, allow_pickle=True)
    microstate_sequence = npz_data['microstate_sequence']

    # Create subfolder if not exist
    subfolder_path = recurrent_sentences_raw_path
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
    assert len(segments) == len(repetition)
    print(f'File {file} processed. Save in {subfolder_path}')
    np.save(os.path.join(subfolder_path, f'{file.replace(".npz", "")}_seg.npy'),
            np.array(segments, dtype='object'), allow_pickle=True)
    np.save(os.path.join(subfolder_path, f'{file.replace(".npz", "")}_repetitions.npy'),
            np.array(repetition, dtype='object'), allow_pickle=True)

# === Step 2: Convert to plain text ===
print("=== Step 2: Convert to plain text ===")
path_current = dataset_base_path
if not os.path.isdir(path_current):
    raise Exception()

files = os.listdir(path_current)
for file in files:
    if not file.endswith('.npz'):
        continue

    seg_file = os.path.join(recurrent_sentences_raw_path,  file.replace(".npz", "") + "_seg.npy")
    repetitions_file = os.path.join(recurrent_sentences_raw_path,  file.replace(".npz", "") + "_repetitions.npy")

    seg_data = np.load(seg_file, allow_pickle=True) + 1
    repetitions_data = np.load(repetitions_file, allow_pickle=True)

    subfolder_path = plain_text_folder
    os.makedirs(subfolder_path, exist_ok=True)

    out_seg_data_plain_text = os.path.join(subfolder_path, file.replace(".npz", "") + "_seg.txt")
    out_repetitions_data_plain_text = os.path.join(subfolder_path, file.replace(".npz", "") + "_repetitions.txt")

    with open(out_seg_data_plain_text, 'w') as seg_file_io, open(out_repetitions_data_plain_text, 'w') as repetitions_file_io:
        assert len(seg_data) == len(repetitions_data)
        for i in range(len(repetitions_data)):
            assert len(seg_data[i]) == len(repetitions_data[i])
            seg_line = ' '.join(map(str, seg_data[i]))
            rep_line = ' '.join(map(str, repetitions_data[i]))
            seg_file_io.write(seg_line + '\n')
            repetitions_file_io.write(rep_line + '\n')