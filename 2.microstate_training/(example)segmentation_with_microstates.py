import os, sys
sys.path.append("../lib")
sys.path.append("../third_parts/microstate_lib/code")
sys.path.append("../")
from dataset.preprocessing import PreprocessingController
import numpy as np
import mne
from datetime import datetime
from dataset.dataset import *
import eeg_recording
import argparse
import json
from data_utils import corr_vectors, get_gfp_peaks


global start_time
global end_time
start_time = -1
end_time = -1


def get_arguments(record_configuration):
    dataset_base_path = record_configuration['extraction_process']['dataset_base_path']
    dataset_name = record_configuration['extraction_process']['database_name']
    record_indexes = record_configuration['indexes']
    preprocessing_pipeline = record_configuration['preprocessings']['pipeline']
    post_merge_pipeline =  record_configuration['preprocessings']['post_merge_pipeline']
    
    # microstate_search_range = (record_configuration['extraction_process'].get('number-microstate-least', 4), 
    #     record_configuration['extraction_process'].get('number-microstate-most', 4))
    # n_iters = record_configuration['extraction_process'].get('kmeans_iterations', 200)
    # stop_delta_threshold = record_configuration['extraction_process'].get('stop_threshold', 0.025)
    # store_4_microstates = record_configuration['extraction_process'].get('store_microstates_n4', True)
    save_preprocessed_data = record_configuration['extraction_process'].get('store_preprocessed', True)
    # save_segmentation = record_configuration['extraction_process'].get("save_segmentation", True)
    load_preprocessed = record_configuration['extraction_process'].get("load_preprocessed", True)
    microstate_file_path =  record_configuration['microstates']['microstate_file_path']

    store_base_path = record_configuration['extraction_process']['store-path']
    return (
        dataset_base_path,
        dataset_name,
        record_indexes,
        preprocessing_pipeline,
        post_merge_pipeline,
        store_base_path,
        save_preprocessed_data,
        load_preprocessed,
        microstate_file_path
    )
def begin_timing():
    global start_time
    start_time = datetime.now()
    
def end_timing():
    global end_time
    end_time = datetime.now()
    
def report_execution_time(event = ""):
    end_timing()
    print('[%s] Time Consumption: {}'.format(event, end_time - start_time))


## ------------------------------- MAIN PART ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-dic", "--database_index_configuration", 
    default="./configs/config-all-person-microstate-retained.json")
args = parser.parse_args()

with open(args.database_index_configuration) as f: 
    data = f.read() 
    record_configuration = json.loads(data)
    f.close()

(
        dataset_base_path,
        dataset_name,
        record_indexes,
        preprocessing_pipeline,
        post_merge_pipeline,
        store_base_path,
        save_preprocessed_data,
        load_preprocessed,
        microstate_file_path
) = get_arguments(record_configuration)

dataset_facade = EEGDatasetFacade(dataset_base_path=dataset_base_path)
dataset = dataset_facade(dataset_name)
if not os.path.exists(store_base_path):
    os.makedirs(store_base_path)

montage_kind = "standard_1020"
montage = mne.channels.make_standard_montage(montage_kind)

preprocessed_file_prefix = \
    record_configuration['extraction_process'].get('preprocessed_file_prefix', '[preprocessed_prep_asr]')

for person_index in record_indexes:
    # PART I : preprocessing
    print(f"Train microstates for person {person_index}")
    record_index_list = record_indexes[person_index]
    expect_preprocessed_file_path = os.path.join(store_base_path, 
            f'{preprocessed_file_prefix}p{person_index}.edf')
    
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
            
            # apply preprocessing pipeline
            for index, preprocessing_pipeline_item in enumerate(preprocessing_pipeline):
                print(f"[Preprocessing {index}: {slice_begin // block_size + 1}/{int(np.ceil(data_count / block_size))}]... name = {preprocessing_pipeline_item[0]}")
                preprocessing_name = preprocessing_pipeline_item[0]
                preprocessing_arguments = preprocessing_pipeline_item[1]
                PreprocessingController.preprocessing(data, preprocessing_name, preprocessing_arguments)
                print(f"End of [Preprocessing {index}: {slice_begin // block_size + 1}/{int(np.ceil(data_count / block_size))}]... name = {preprocessing_pipeline_item[0]}")
            
            results.append(data)
        
        data = mne.concatenate_raws(results)
        
        del results
        
        for index, postprocessing_pipeline_item in enumerate(post_merge_pipeline):
            print(f"[Post Merging Preprocessing {index}: {slice_begin // block_size + 1}/{int(np.ceil(data_count / block_size))}]... name = {postprocessing_pipeline_item[0]}")
            postprocessing_name = postprocessing_pipeline_item[0]
            postprocessing_arguments = postprocessing_pipeline_item[1]
            PreprocessingController.preprocessing(data, postprocessing_name, postprocessing_arguments)
        if save_preprocessed_data:
                mne.export.export_raw(expect_preprocessed_file_path, data, overwrite=True)
        
    
    # PART II: train microstates
    recording = eeg_recording.SingleSubjectRecording("0", data)
    

    print(f"Begin segment signal with microstates. Result will save in '{store_base_path}'")
    
    # GEV of training result of previous amount of microstates. 
    maps = np.load(microstate_file_path)
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
    print(gev, gev_gfp)
    # store the best set of microstates, i.e., the last one.
    