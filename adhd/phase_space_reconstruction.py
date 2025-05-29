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

binary_path = '../lib/pcfg-cky-inside-outside/bin/phase_convert'

for l in L:
    path_current_l = os.path.join(dataset_base_path, l)
    if not os.path.isdir(path_current_l):
        continue

    files = os.listdir(path_current_l)
    for file in files:
        if not file.endswith('.npz'):
            continue

        full_seg_path = os.path.join(plain_text_folder, l, file.replace('.npz', "_seg_plain_text.txt"))
        full_repetitions_path = os.path.join(plain_text_folder, l, file.replace('.npz', "_repetitions_plain_text.txt"))


        full_out_seg_path = os.path.join(converted_sentences_path, l, file.replace('.npz', "_seg_plain_text.txt"))
        full_out_repetitions_path = os.path.join(converted_sentences_path, l, file.replace('.npz', "_repetitions_plain_text.txt"))

        # Create subfolder if not exist
        subfolder_path = os.path.join(converted_sentences_path, l)
        os.makedirs(subfolder_path, exist_ok=True)
        
                # Create tmp_config.yaml for the conversion
        tmp_configuration = f'''\
        phase_convert:
        grammar_file: "../data/pcfg/grammar.pcfg"
        input: "{full_seg_path}"
        output: "{full_out_seg_path}"
        '''

        with open("tmp_config.yaml", "w") as f:
            f.write(tmp_configuration)

        command = ['python', binary_path, 'tmp_config.yaml']
        subprocess.run(command)
        os.remove('tmp_config.yaml')