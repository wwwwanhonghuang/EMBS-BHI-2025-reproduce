import sys
sys.path.append("../lib")
sys.path.append("../third_parts/microstate_lib/code")
sys.path.append("../")
import numpy as np
import mne
from dataset.dataset import *
import argparse
from data_utils import corr_vectors, get_gfp_peaks

parser = argparse.ArgumentParser()
parser.add_argument("--record_file_path", type=str, required=True)
parser.add_argument("--microstate_file_path", type=str, required=True)
parser.add_argument("--sequence_file_save_path", type=str, required=True)

args = parser.parse_args()

record_file_path = args.record_file_path
microstate_file_path = args.microstate_file_path

print(f"Load EEG record from file {record_file_path}")
data = mne.io.read_raw(record_file_path).get_data()

# GEV of training result of previous amount of microstates. 
maps = np.load(microstate_file_path)
print(f"Topomaps shape = {maps.shape}")
(peaks, gfp_curve) = get_gfp_peaks(
        data,
        min_peak_dist=2,
        smoothing=None,
        smoothing_window=100,
)

activation = maps.dot(data) 
segmentation = np.argmax(np.abs(activation), axis=0)
map_corr = corr_vectors(data, maps[segmentation].T)
gfp_corr = corr_vectors(data[:, peaks], maps[segmentation[peaks]].T)
gfp_sum_sq = np.sum(gfp_curve ** 2)
peaks_sum_sq = np.sum(data[:, peaks].std(axis=0) ** 2)

gev = sum((gfp_curve * map_corr) ** 2) / gfp_sum_sq

gev_gfp = (
    sum((data[:, peaks].std(axis=0) * gfp_corr) ** 2) / peaks_sum_sq
)

print(f'gev = {gev}, gev_gfp = {gev_gfp}')

np.save(args.sequence_file_save_path, segmentation)

print(f"Microstate sequence have been saved in {args.sequence_file_save_path}")
