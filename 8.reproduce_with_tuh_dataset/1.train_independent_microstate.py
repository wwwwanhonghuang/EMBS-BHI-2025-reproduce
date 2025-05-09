import os
import mne
import sys
sys.path.append("..")
sys.path.append("../lib")
sys.path.append("../third_parts/microstate_lib/code")
import eeg_recording
from tqdm import tqdm
import numpy as np

dev_path = "/data1/tuh_eeg_seizure_corpus/edf/dev"

def read_all_dataset_info(dataset_type="dev"):
    base_paths = {
        'dev': dev_path 
    }
    
    if dataset_type not in base_paths:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    base_path = base_paths[dataset_type]
    files = []

    def _find_files_recursively(path):
        if os.path.isfile(path) and os.path.splitext(path)[1] == ".edf":
            if os.path.exists(path[:-4] + ".csv"):
                files.append(path)
            else:
                print(f"Warning: csv record not exist for {path}, skipping.")
            return
        
        if os.path.isdir(path):
            for file in os.listdir(path):
                _find_files_recursively(os.path.join(path, file))

    _find_files_recursively(base_path)
    return files  # Return the collected file list

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
    
dev_files = read_all_dataset_info()

topo_maps = []
retain_indexes = []

gev_threshold = 0.5
dispose = 0

for index, file in enumerate(tqdm(dev_files)):
    raw = mne.io.read_raw(file, preload=True, verbose = 'WARNING')
    normalize_to_10_20_montage(raw)
    recording = eeg_recording.SingleSubjectRecording("0", raw)
    recording.run_latent_kmeans(4, use_gfp = True)
    print(recording.gev_tot)

    if recording.gev_tot >= gev_threshold:
        topo_maps.append(recording.latent_maps)
        print(recording.latent_maps.shape)
        retain_indexes.append(index)
    else:
        dispose += 1
        print(f"Warining: dispose record {file}, due to the total gev {recording.gev_tot} < {gev_threshold}. Current disposing rate = {dispose / (index + 1)}")
        
topo_maps_array = np.empty(len(topo_maps), dtype=object)
for i, item in enumerate(topo_maps):
    topo_maps_array[i] = np.asarray(item, dtype=object)

np.save("data/retain_indexes.npy", retain_indexes, allow_pickle=True)
np.save("data/topo_maps.npy", topo_maps_array, allow_pickle=True)
