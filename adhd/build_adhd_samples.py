import os
import sys
import numpy as np
import mne
from collections import defaultdict
import argparse
import random
from tqdm import tqdm  # for progress bar
import ast
import re
sys.path.append("../third_parts/microstate_lib/code")
sys.path.append("../lib")
sys.path.append("../")
import eeg_recording

from data_utils import corr_vectors, get_gfp_peaks

def parse_list(arg):
    return ast.literal_eval(arg)


def min_max_nor(mne_raw):
    """Normalize data to [0,1] range per channel"""
    data = mne_raw.get_data()
    min_data = np.min(data, axis=1, keepdims=True)
    max_data = np.max(data, axis=1, keepdims=True)
    minmax_data = (data - min_data) / (max_data - min_data + 1e-10)  # avoid division by zero
    return mne.io.RawArray(minmax_data, mne_raw.info)


def produce_microstates_and_sequence(sample_id, sample_file):
    """Process sample files and compute microstates"""
    try:
        # Load and concatenate files
        raw = mne.io.read_raw_fif(os.path.join(args.preprocessed_dataset_base_path, sample_file)) 
        data = raw
        # Preprocessing
        data.load_data()
        data.filter(l_freq=0, h_freq=40)
        data = data.set_eeg_reference("average")
        data = min_max_nor(data)
        
        # Save results
        save_path = save_base_path
        os.makedirs(save_path, exist_ok=True)
            
        save_file = os.path.join(save_path,
            f"A{sample_id}.npz")
        
        recording = eeg_recording.SingleSubjectRecording(sample_id, data)
        recording.run_latent_kmeans(n_states=args.n_states, use_gfp=False)
        print(f'total_gev = {recording.gev_tot}')

        np.savez(save_file,
                microstate_sequence=recording.latent_segmentation,
                microstates=recording.latent_maps,
                gev=recording.gev_tot, sample_id=sample_id)
        
        return True
    except Exception as e:
        print(f"Error processing {sample_id}: {str(e)}")
        print()
        return False
    
# Argument parsing
parser = argparse.ArgumentParser(description='Generate microstate samples from ADHD dataset')
parser.add_argument("--preprocessed_dataset_base_path", 
                   default="./data/adhd_control_preprocessed", 
                   type=str)


parser.add_argument("--n_states", 
                   default=4, 
                   type=int,
                   help="Number of microstates to compute")
parser.add_argument("--seed", 
                   default=42, 
                   type=int,
                   help="Random seed for reproducibility")

parser.add_argument("--dataset_base_path", default="./data/adhd_microstate_dataset_samples/", type=str)


args = parser.parse_args()

dataset_base_path = args.dataset_base_path

sample_dataset_base_path = os.path.join(dataset_base_path, "..", "adhd_microstate_dataset_samples", "l")


adhd_data_set_path = os.path.join(dataset_base_path, "..", "adhd_control_preprocessed")
files = os.listdir(adhd_data_set_path)
for file in files:
    if not file.endswith('.npz'):
        continue

args = parser.parse_args()
save_base_path = os.path.join(dataset_base_path, "..", "syntax_analysis_data", "adhd")
# Ensure save directory exists
os.makedirs(save_base_path, exist_ok=True)


# Group files by prefix (A/C)
grouped_list = defaultdict(list)
for item in os.listdir(args.preprocessed_dataset_base_path):
    if item.endswith('.fif'):
        key = item[0]  # 'A' or 'C'
        number = f"{int(item[1:-4]):02d}"
        grouped_list[key].append(f"{key}{number}.fif")

adhd_samples = grouped_list['A']
total_files = len(adhd_samples)


pbar = tqdm(total=total_files, desc="Generating samples")

for sample_id in range(total_files):
    if produce_microstates_and_sequence(sample_id, adhd_samples[sample_id]):
        pbar.update(1)
    else:
        raise Exception()

pbar.close()
print(f"Finished generating samples. Totally {total_files} samples.")
