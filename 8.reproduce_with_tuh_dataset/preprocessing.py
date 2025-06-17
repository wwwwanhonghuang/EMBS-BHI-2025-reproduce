import os
import mne
import sys
import argparse
import numpy as np
from tqdm import tqdm

# Configure paths
sys.path.extend(["..", "../lib", "../third_parts/microstate_lib/code"])

def setup_args():
    parser = argparse.ArgumentParser(description='EEG Data Preprocessing')
    parser.add_argument("--dev_path", default="/data1/tuh_eeg_seizure_corpus/edf/dev",
                      help="Path to development dataset")
    parser.add_argument("--output_path", default="./data/preprocessed_data",
                      help="Output directory for processed data")
    return parser.parse_args()

def find_edf_files(base_path):
    """Recursively find all EDF files with corresponding CSV files."""
    valid_files = []
    
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.edf'):
                edf_path = os.path.join(root, file)
                csv_path = edf_path[:-4] + '.csv'
                
                if os.path.exists(csv_path):
                    valid_files.append(edf_path)
                else:
                    print(f"Warning: CSV record missing for {edf_path}")
    
    return valid_files


def normalize_to_10_20_montage(raw):
    def _check_monopolar(ch_names):
        propable_reference_name = ['REF', 'LE', 'RE']
        if(not any (raw.ch_names[0].find(name) >= 0 for name in propable_reference_name)):
            return False
        return True
    if not _check_monopolar(raw.ch_names):
        print(f"Error: {raw.ch_names} may not in a monopolar reference.")
        raise ValueError
    standard_10_20_names = [
        'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ', 'OZ'
    ]
    ch_names = raw.ch_names
    ch_name_mapping = {
        name: next((standard_name for standard_name in standard_10_20_names if standard_name+"-" in name and name[:3] == "EEG"), None)
        for name in ch_names
    }
    # Filter out channels that do not have a standard 10-20 name
    filtered_ch_names = {name: standard_name for name, standard_name in ch_name_mapping.items() if standard_name is not None}

    raw.drop_channels([ch for ch in raw.ch_names if ch not in ch_name_mapping or ch_name_mapping[ch] is None])
    #print(filtered_ch_names, raw.ch_names)
    # Rename the channels in the raw object
    raw.rename_channels(filtered_ch_names)

    channel_rename_mapping = {
        'FP1': 'Fp1',
        'FP2': 'Fp2',
        'FZ': 'Fz',
        'CZ': 'Cz',
        'PZ': 'Pz',
        'OZ': 'Oz'
    }
    
    channel_rename_mapping = {source_name : channel_rename_mapping[source_name] \
        for source_name in channel_rename_mapping if source_name in raw.ch_names}

    # Rename the channels in the raw object
    raw.rename_channels(channel_rename_mapping)

    raw.set_montage("standard_1020")

def normalize_data(raw):
    """Apply z-score normalization per channel."""
    data = raw.get_data()
    normalized = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
    raw._data = normalized
    return raw

def process_file(edf_path, output_dir):
    """Process single EDF file."""
    try:
        raw = mne.io.read_raw(edf_path, preload=True, verbose='WARNING')
        raw.filter(0.5, 110)
        normalize_to_10_20_montage(raw)
        normalize_data(raw)
        
        # Save processed file
        base_name = os.path.basename(edf_path)
        output_path = os.path.join(output_dir, base_name)
        output_path = output_path[:-3] + "fif"
        raw.save(output_path, overwrite=True)
    except Exception as e:
        print(f"Error processing {edf_path}: {str(e)}")


def main():
    args = setup_args()
    os.makedirs(args.output_path, exist_ok=True)
    
    edf_files = find_edf_files(args.dev_path)
    print(f"Found {len(edf_files)} EDF files to process")
    
    for file in tqdm(edf_files, desc="Processing EEG files"):
        process_file(file, args.output_path)

if __name__ == "__main__":
    main()