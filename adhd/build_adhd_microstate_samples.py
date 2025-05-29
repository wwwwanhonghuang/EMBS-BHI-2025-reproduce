import os
import sys
import numpy as np
import mne
from collections import defaultdict
import argparse
import random
from tqdm import tqdm  # for progress bar
import ast
sys.path.append("../third_parts/microstate_lib/code")
import eeg_recording
import hashlib

def parse_list(arg):
    return ast.literal_eval(arg)


def min_max_nor(mne_raw):
    """Normalize data to [0,1] range per channel"""
    data = mne_raw.get_data()
    min_data = np.min(data, axis=1, keepdims=True)
    max_data = np.max(data, axis=1, keepdims=True)
    minmax_data = (data - min_data) / (max_data - min_data + 1e-10)  # avoid division by zero
    return mne.io.RawArray(minmax_data, mne_raw.info)

def generate_sample_id(indexes):
    """Generate unique ID from sorted indexes"""
    return '_'.join(f"{i:02d}" for i in sorted(indexes))

def produce_microstates_and_sequence(sample_id, l, sample_files):
    """Process sample files and compute microstates"""
    try:
        # Load and concatenate files
        raws = [mne.io.read_raw_fif(os.path.join(args.preprocessed_dataset_base_path, f)) 
                for f in sample_files]
        data = mne.concatenate_raws(raws)
        
        # Preprocessing
        data.load_data()
        data.filter(l_freq=0, h_freq=40)
        data = data.set_eeg_reference("average")
        data = min_max_nor(data)
        
        # Microstate analysis
        recording = eeg_recording.SingleSubjectRecording(sample_id, data)
        recording.run_latent_kmeans(n_states=args.n_states, use_gfp=False)
        
        # Save results
        save_path = os.path.join(args.save_base_path, str(l))
        os.makedirs(save_path, exist_ok=True)
            
        save_file = os.path.join(save_path,
            f"{hashlib.md5(sample_id.encode()).hexdigest()[:8]}_gev_{recording.gev_tot:.4f}.npz")
        np.savez(save_file,
                microstate_sequence=recording.latent_segmentation,
                microstates=recording.latent_maps,
                gev=recording.gev_tot, sample_id=sample_id)
        
        return True
    except Exception as e:
        print(f"Error processing {sample_id}: {str(e)}")
        return False
    
# Argument parsing
parser = argparse.ArgumentParser(description='Generate microstate samples from ADHD dataset')
parser.add_argument("--preprocessed_dataset_base_path", 
                   default="./data/adhd_control_preprocessed", 
                   type=str)
parser.add_argument("--save_base_path", 
                   default="./data/adhd_microstate_dataset_samples", 
                   type=str)
parser.add_argument("--K", 
                   default=100, 
                   type=int,
                   help="Number of sample sets to generate")
parser.add_argument("--n_states", 
                   default=4, 
                   type=int,
                   help="Number of microstates to compute")
parser.add_argument("--seed", 
                   default=42, 
                   type=int,
                   help="Random seed for reproducibility")

parser.add_argument("--L", type=parse_list, default = [1,2,5,10,20,30])



args = parser.parse_args()

# Ensure save directory exists
os.makedirs(args.save_base_path, exist_ok=True)


# Group files by prefix (A/C)
grouped_list = defaultdict(list)
for item in os.listdir(args.preprocessed_dataset_base_path):
    if item.endswith('.fif'):
        key = item[0]  # 'A' or 'C'
        number = f"{int(item[1:-4]):02d}"
        grouped_list[key].append(f"{key}{number}.fif")

control_samples = grouped_list['C']
total_files = len(control_samples)

# Set random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)

L = args.L
print(f"L values: {L}")

for l in L:
    generated_samples = set()
    pbar = tqdm(total=args.K, desc="Generating samples")

    while len(generated_samples) < args.K:
        sample_size = l
        indexes = random.sample(range(total_files), sample_size)
        
        sample_id = generate_sample_id(indexes)
        if sample_id in generated_samples:
            continue
            
        sample_files = [control_samples[i] for i in indexes]
        if produce_microstates_and_sequence(sample_id, l, sample_files):
            generated_samples.add(sample_id)
            pbar.update(1)

pbar.close()
print(f"Finished generating samples for all L values. Totally {len(L) * args.K} samples.")
