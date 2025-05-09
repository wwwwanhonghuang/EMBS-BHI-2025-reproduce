import os, sys
sys.path.append("../lib")
sys.path.append("../third_parts/microstate_lib/code")
sys.path.append("../")
import numpy as np
import mne
import argparse
import json
from data_utils import corr_vectors, get_gfp_peaks


global start_time
global end_time
start_time = -1
end_time = -1



def get_arguments(record_configuration):
    raw_file_path = record_configuration['record']['raw_file_path']
    
    microstate_file_path =  record_configuration['microstates']['microstate_file_path']

    store_base_path = record_configuration['storage']['store-path']
    
    store_file_name = record_configuration['storage']['store-store_file_name']
    return (
        raw_file_path,
        microstate_file_path,
        store_base_path,
        store_file_name
    )
    

parser = argparse.ArgumentParser()
parser.add_argument("-dic", "--database_index_configuration", 
    default="./configs/config-segmentation.json")

args = parser.parse_args()

with open(args.database_index_configuration) as f: 
    data = f.read() 
    record_configuration = json.loads(data)
    f.close()

(
    raw_file_path,
    microstate_file_path,
    store_base_path,
    store_file_name
) = get_arguments(record_configuration)


montage_kind = "standard_1020"
montage = mne.channels.make_standard_montage(montage_kind)

print(f"Begin segment signal with microstates.")

# GEV of training result of previous amount of microstates. 
pre_gev_tot = 0
maps = np.load(microstate_file_path)
data = mne.io.read_raw(raw_file_path, preload=True).get_data()
data_min = np.min(data)
data_max = np.max(data)
data = (data - data_min) / (data_max - data_min)
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

# store the best set of microstates, i.e., the last one.
if not os.path.exists(store_base_path):
    os.makedirs(store_base_path)
np.save(os.path.join(store_base_path, store_file_name), segmentation)