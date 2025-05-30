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

from lib.dataset.dataset import *
from lib.dataset.experiment_utils import to_segment_sequence


def get_filename(path):
    return os.path.basename(path)

def read_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        data = [[int(e.strip()) for e in line.strip().split(" ")] for line in lines]
        f.close()
        return data
# Argument parsing
parser = argparse.ArgumentParser(description='Generate microstate samples from ADHD dataset')
parser.add_argument("--dataset_base_path", default="./data/adhd_microstate_dataset_samples", type=str)
args = parser.parse_args()

dataset_base_path = args.dataset_base_path
plain_text_folder = os.path.join(dataset_base_path, "..", "plain_text_segments")

assert os.path.exists(plain_text_folder)

converted_sentences_path = os.path.join(dataset_base_path, "..", "sentence_converted")
os.makedirs(converted_sentences_path, exist_ok=True)

L = os.listdir(dataset_base_path)

binary_path = '../lib/pcfg-cky-inside-outside/bin/phase_convert'

pbar = tqdm(total=sum([len(os.listdir(os.path.join(dataset_base_path, l))) for l in L]), desc="Converting samples")
delay = 2
for l in L:
    path_current_l = os.path.join(dataset_base_path, l)
    if not os.path.isdir(path_current_l):
        continue

    files = os.listdir(path_current_l)
    for file in tqdm(files):
        if not file.endswith('.npz'):
            continue

        full_seg_path = os.path.join(plain_text_folder, l, file.replace('.npz', "_seg.txt"))
        full_repetitions_path = os.path.join(plain_text_folder, l, file.replace('.npz', "_repetitions.txt"))


        full_out_seg_path = os.path.join(converted_sentences_path, l, file.replace('.npz', "_seg_text.txt"))
        full_out_repetitions_path = os.path.join(converted_sentences_path, l, file.replace('.npz', "_repetitions.txt"))

        # Create subfolder if not exist
        subfolder_path = os.path.join(converted_sentences_path, l)
        os.makedirs(subfolder_path, exist_ok=True)
        seg_data = read_data(full_seg_path)
        repetition_data = read_data(full_repetitions_path)
        full_out_repetitions_io = open(full_out_repetitions_path, 'w')
        full_out_seg_io = open(full_out_seg_path, 'w')

        assert len(seg_data) == len(repetition_data)
        for i in range(len(repetition_data)):
            rep_line_data = repetition_data[i]
            seg_line_data = seg_data[i]
            assert len(rep_line_data) == len(seg_line_data)
            
            converted_repetition_line_data = np.zeros((len(rep_line_data) - delay + 1), dtype=int)
            time_delayed_seg_data = np.zeros((len(seg_line_data) - delay + 1), dtype=int)
            
            for begin in range(delay - 1, len(seg_line_data)):
                s = 0
                for j in range(0, delay):
                    s *= 4;
                    s += int(seg_line_data[begin - j] - 1)
                
                time_delayed_seg_data[begin - delay + 1] = s + 1
            
            for j in range(delay - 1, len(converted_repetition_line_data) + 1):
                # TODO: optimize this utilize dynamical programming.
                converted_repetition_line_data[j - 1] = sum(rep_line_data[j - delay + 1: j + 1]) # rep_line_data[j] + rep_line_data[j - 1]
            rep_line = ' '.join(map(str, converted_repetition_line_data))
            seg_line = ' '.join(map(str, time_delayed_seg_data))

            full_out_repetitions_io.write(rep_line + '\n')
            full_out_seg_io.write(seg_line + '\n')
            

        pbar.update(1)

