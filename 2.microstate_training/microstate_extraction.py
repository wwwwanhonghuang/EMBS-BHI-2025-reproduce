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
    microstate_search_range = (record_configuration['extraction_process'].get('number-microstate-least', 4), 
        record_configuration['extraction_process'].get('number-microstate-most', 4))
    n_iters = record_configuration['extraction_process'].get('kmeans_iterations', 200)
    stop_delta_threshold = record_configuration['extraction_process'].get('stop_threshold', 0.025)
    store_4_microstates = record_configuration['extraction_process'].get('store_microstates_n4', True)
    save_preprocessed_data = record_configuration['extraction_process'].get('store_preprocessed', True)
    save_segmentation = record_configuration['extraction_process'].get("save_segmentation", True)
    load_preprocessed = record_configuration['extraction_process'].get("load_preprocessed", False)
    store_base_path = record_configuration['extraction_process']['store-path']
    return (
        dataset_base_path,
        dataset_name,
        record_indexes,
        preprocessing_pipeline,
        post_merge_pipeline,
        microstate_search_range,
        n_iters,
        stop_delta_threshold,
        store_4_microstates,
        save_preprocessed_data,
        save_segmentation,
        load_preprocessed,
        store_base_path,
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

def store(maps, segmentation, gev, preprocessing_desc, person_id):
    n_states = maps.shape[0]
    save_map_file_name = f"[{preprocessing_desc}]person_{person_id}_states{n_states}_gev_{gev}.npy"

    np.save(os.path.join(store_base_path, save_map_file_name), maps)
    if save_segmentation:
        save_segmentation_file_name = f"[seg-{preprocessing_desc}]person_{person_id}_states{n_states}_gev_{gev}.npy"
        np.save(os.path.join(store_base_path, save_segmentation_file_name), segmentation)

def split_data(data, record_ids, sampling_frequency):
    sampling_frequency = int(sampling_frequency)
    
    '''
    Seizure interval annocations of the aggregated data. 
    In global time stamp.
    '''
    seizures = dataset.get_seizure_ranges_in_time_offsets(record_ids)
    
    length_of_pre_epileptic_zone = args.pre_epileptic_zone
    seizures = sorted(seizures, key=lambda x: x[0])
    assert(all(seizure[1] <= data.shape[0] for seizure in seizures))

    splitted_data = {
        'seizure': [],
        'pre-epileptic':[],
        'normal': []
    }

    # Add seizure segmentations
    for seizure_annotation in seizures:
        splitted_data['seizure'].append(data[seizure_annotation[0]: seizure_annotation[1] + 1])
        print(f'added seizure segment {seizure_annotation[0]}:{seizure_annotation[1] + 1}')
    
    previous_seizure_ending_position = 0

    # Add pre-epileptic and normal segmentations
    for seizure_annotation in seizures:

        # last seizure end is the normal part's beginning.
        normal_part_begin = previous_seizure_ending_position
        pre_epileptic_end = seizure_annotation[0]

        # 'length_of_pre_epileptic_zone' secs back from current seizure onset.
        pre_epileptic_begin = max(seizure_annotation[0] - sampling_frequency * length_of_pre_epileptic_zone, previous_seizure_ending_position)
        
        # current pre-epileptic beginning is the normal part's ending.
        normal_part_end = pre_epileptic_begin

        ## layout: | seizure_{last} | normal | pre-epileptic | seizure_{current}

        if(normal_part_end > normal_part_begin):
            splitted_data['normal'].append(data[normal_part_begin: normal_part_end])
            print(f'added normal segment {normal_part_begin}:{normal_part_end}, {data.shape}')

        if(pre_epileptic_end > pre_epileptic_begin):
            splitted_data['pre-epileptic'].append(data[pre_epileptic_begin: pre_epileptic_end])
            print(f'added pre-epileptic segment {pre_epileptic_begin}:{pre_epileptic_end}')

        # update previous_seizure_ending_position
        previous_seizure_ending_position = seizure_annotation[1] + 1
    
    splitted_data['normal'].append(data[previous_seizure_ending_position: data.shape[0]])
    
    # Check sum of segment lengths
    seizure_length = sum([len(segment) for segment in splitted_data['seizure']])
    normal_length = sum([len(segment) for segment in splitted_data['normal']])
    pre_epileptic_length = sum([len(segment) for segment in splitted_data['pre-epileptic']])
    total_length = sum([seizure_length, normal_length, pre_epileptic_length])
    assert total_length == data.shape[0], \
        f"Mismatch in data length: {data.shape[0]} != {total_length}"

    print(f"seizures n_segments = {sum([len(segment) for segment in splitted_data['seizure']])}, normal n_segments = {sum([len(segment) for segment in splitted_data['normal']])}, pre-epileptic n_segments = {sum([len(segment) for segment in splitted_data['pre-epileptic']])}")
    print("Checks data length ==  total length of splitted fragments...Passed.")

    return splitted_data

## ------------------------------- MAIN PART ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-dic", "--database_index_configuration", 
    default="./configs/config-all-person-microstate-dev.json")
parser.add_argument("--pre_epileptic_zone", default=60 * 5)
parser.add_argument("--no_normal_only", action="store_true")
parser.add_argument("--force_repreprocessing", action="store_true")

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
        microstate_search_range,
        n_iters,
        stop_delta_threshold,
        store_4_microstates,
        save_preprocessed_data,
        save_segmentation,
        load_preprocessed,
        store_base_path
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
    
    if not args.force_repreprocessing and load_preprocessed and os.path.exists(expect_preprocessed_file_path):
        print(f"Load preprocessed data...")
        data = mne.io.read_raw(expect_preprocessed_file_path)
    else:
        if load_preprocessed and not os.path.exists(expect_preprocessed_file_path):
            print(f"Cannot find preprocessed Raw Data file from path {expect_preprocessed_file_path}. Try repreprocessing raw data.")
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
    
    # PART II: Retain only normal part
    if not args.no_normal_only:
        print(f'In training microstate in normal area only mode.')
        sampling_freq = int(data.info['sfreq'])
        splitted_data = split_data(data.get_data().T, record_index_list, sampling_freq)
        normal_areas = splitted_data['normal'] # a list of ndarray, shaoe = (T_i, n_channel)
        merged_normal_area = np.concatenate(normal_areas, axis=0)
        print(f'Normal area total length = {merged_normal_area.shape[0]}, Pre-epileptic length = {args.pre_epileptic_zone * sampling_freq}')
        raw_normal = mne.io.RawArray(merged_normal_area.T, data.info)  # MNE expects shape (n_channels, n_times)

    
    # PART III: train microstates
    if not args.no_normal_only:
        print(f'Build microstate training object with normal area only MNE raw data')
        recording = eeg_recording.SingleSubjectRecording("0", raw_normal)
    else:
        recording = eeg_recording.SingleSubjectRecording("0", data)

    print(f"Begin training microstates. Result will save in '{store_base_path}'")
    
    print(f" -- Search Microstate Amount from {microstate_search_range[0]} to {microstate_search_range[1]}")
    
    # GEV of training result of previous amount of microstates. 
    pre_gev_tot = 0
    
    # Train microstate sets with various numbers.
    for n_states in range(microstate_search_range[0], microstate_search_range[1] + 1):
        print(f"Begin training {n_states} microstates")
        recording.run_latent_kmeans(n_states = n_states, use_gfp = True, n_inits = n_iters)
        
        if recording.latent_maps is None: # No result.
            continue
        
        current_gev_tot = recording.gev_tot
        print(f'previous gev_tot = {pre_gev_tot}, current_gev_tot = {current_gev_tot}')
        
        # Early stop training larger amount of microstates,
        # if GEV increment is smaller than threshold
        delta = current_gev_tot - pre_gev_tot
        if delta < stop_delta_threshold:
            break
        
        # Save size-4 microstate set if expected. 
        if n_states == 4 and store_4_microstates:
            store(recording.latent_maps, recording.latent_segmentation, recording.gev_tot, 
                record_configuration['save_prefix'], person_index)
        pre_gev_tot = current_gev_tot
        print(f" - n_states = {n_states}, gev_tot = {current_gev_tot}. --")
        
    # store the best set of microstates, i.e., the last one.
    store(recording.latent_maps, recording.latent_segmentation, 
        recording.gev_tot, record_configuration['save_prefix'], person_index)
