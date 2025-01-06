import numpy as np
import mne
import os, sys
from datetime import datetime
sys.path.append("../lib")
sys.path.append("../third_parts/microstate_lib/code")
from dataset.dataset import *
import eeg_recording
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-dic", "--database_index_configuration", default="./configs/config-all-person-microstate.json")
args = parser.parse_args()

with open(args.database_index_configuration) as f: 
        data = f.read() 
        record_configuration = json.loads(data)
        f.close()

dataset_base_path = record_configuration['extraction_process']['dataset_base_path']
dataset_name = record_configuration['extraction_process']['database_name']
record_indexes = record_configuration['indexes']
preprocessing_pipeline = record_configuration['preprocessings']['pipeline']
post_merge_pipeline =  record_configuration['preprocessings']['post_merge_pipeline']
microstate_search_range = (record_configuration['extraction_process'].get('number-microstate-least', 4), 
    record_configuration['extraction_process'].get('number-microstate-most', 4))
n_iters = record_configuration['extraction_process'].get('kmeans_iterations', 200)
stop_delta_threshold = record_configuration['extraction_process'].get('stop_threshold', 0.025)
store_4_microstates = record_configuration['extraction_process'].get('store_microstates_n4', True)
save_preprocessed_data = record_configuration['extraction_process'].get('store_preprocessed', True)
save_segmentation = record_configuration['extraction_process'].get("save_segmentation", True)
load_preprocessed = record_configuration['extraction_process'].get("load_preprocessed", True)

dataset_facade = EEGDatasetFacade(dataset_base_path=dataset_base_path)
dataset = dataset_facade(dataset_name)
store_base_path = record_configuration['extraction_process']['store-path']
if not os.path.exists(store_base_path):
    os.makedirs(store_base_path)
    
global start_time
global end_time
start_time = -1
end_time = -1

def begin_timing():
    global start_time
    start_time = datetime.now()
    
def end_timing():
    global end_time
    end_time = datetime.now()
    
def report_execution_time(event = ""):
    end_timing()
    print('[%s] Time Consumption: {}'.format(event, end_time - start_time))

def store(maps, segmentation, gev, preprocessing_desc, person_id):
    n_states = maps.shape[0]
    save_map_file_name = f"[{preprocessing_desc}]person_{person_id}_states{n_states}_gev_{gev}.npy"

    np.save(os.path.join(store_base_path, save_map_file_name), maps)
    if save_segmentation:
        save_segmentation_file_name = f"[seg-{preprocessing_desc}]person_{person_id}_states{n_states}_gev_{gev}.npy"
        np.save(os.path.join(store_base_path, save_segmentation_file_name), segmentation)


montage_kind = "standard_1020"
montage = mne.channels.make_standard_montage(montage_kind)
sys.path.append("../")
from dataset.preprocessing import PreprocessingController

preprocessed_file_prefix = \
    record_configuration['extraction_process'].get('preprocessed_file_prefix', '[preprocessed_prep_asr]')


for person_index in record_indexes:
    print(f"Train microstates for person {person_index}")
    record_index_list = record_indexes[person_index]
    expect_preprocessed_file_path = os.path.join(store_base_path, f'{preprocessed_file_prefix}p{person_index}.edf')
    if load_preprocessed and os.path.exists(expect_preprocessed_file_path):
        print(f"Load preprocessed data...")
        data = mne.io.read_raw(expect_preprocessed_file_path)
    else:
        data_count = len(record_index_list)
        results = []
        block_size = 1
        for slice_begin in range(0, data_count, block_size):
            data = dataset.get_merge_mne_data(record_index_list[slice_begin: slice_begin + block_size])
            
            data.rename_channels({ch_name: ch_name.replace("EEG ", "").replace("-Ref", "") for ch_name in data.ch_names})
            
            #! --- preprocessing ---
            for index, preprocessing_pipeline_item in enumerate(preprocessing_pipeline):
                print(f"[Preprocessing {index}: {slice_begin // block_size + 1}/{int(np.ceil(data_count / block_size))}]... name = {preprocessing_pipeline_item[0]}")
                preprocessing_name = preprocessing_pipeline_item[0]
                preprocessing_arguments = preprocessing_pipeline_item[1]
                PreprocessingController.preprocessing(data, preprocessing_name, preprocessing_arguments)
            
            results.append(data)
        data = mne.concatenate_raws(results)
        del results
        
        for index, preprocessing_pipeline_item in enumerate(post_merge_pipeline):
            print(f"[Post Merging Preprocessing {index}: {slice_begin // block_size + 1}/{int(np.ceil(data_count / block_size))}]... name = {preprocessing_pipeline_item[0]}")
            preprocessing_name = preprocessing_pipeline_item[0]
            preprocessing_arguments = preprocessing_pipeline_item[1]
            PreprocessingController.preprocessing(data, preprocessing_name, preprocessing_arguments)
        
        if save_preprocessed_data:
            mne.export.export_raw(expect_preprocessed_file_path, data, overwrite=True)
    
    recording = eeg_recording.SingleSubjectRecording("0", data)

    #! --- microstate training
    print(f"Begin training microstates. Result will save in '{store_base_path}'")
    print(f" -- Search Microstate Amount from {microstate_search_range[0]} to {microstate_search_range[1]}")
    pre_gev_tot = 0
    for n_states in range(microstate_search_range[0], microstate_search_range[1] + 1):
        print(f"Begin Training {n_states} microstates")
        recording.run_latent_kmeans(n_states = n_states, use_gfp = True, n_inits = n_iters)
        if recording.latent_maps is None:
            continue
        current_gev_tot = recording.gev_tot
        print(f'previous gev_tot = {pre_gev_tot}, current_gev_tot = {current_gev_tot}')
        delta = current_gev_tot - pre_gev_tot
        if delta < stop_delta_threshold:
            break
        if n_states == 4 and store_4_microstates:
            store(recording.latent_maps, recording.latent_segmentation, recording.gev_tot, "[prep-asr]", person_index)
        pre_gev_tot = current_gev_tot
        print(f" -- n_states = {n_states}, gev_tot = {current_gev_tot}. --")
        
    store(recording.latent_maps, recording.latent_segmentation, recording.gev_tot, record_configuration['save_prefix'], person_index)