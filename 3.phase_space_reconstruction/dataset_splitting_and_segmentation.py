
import numpy as np
import re
import os, sys
import json
import argparse
from functools import reduce

sys.path.append("../")
sys.path.append("../lib/microstate_lib/code")
from segmentation_module import InfinitePhaseSpaceReonstructionBasedSegmentGenerator, FiniteTimeDelaySegmentGenerator, FiniteTimeDelayEEGSegmentGenerator

from lib.dataset.dataset import *
from lib.dataset.experiment_utils import to_segment_sequence

def build_recurrence_sentences(microstate_data, zone_type: str):
    print(f"Proceeding {zone_type} segments...")
    all_segments = []
    all_repetitions = []
    for data_index, data in enumerate(microstate_data[f"{zone_type}"]):
        segment_generator = FiniteTimeDelaySegmentGenerator(data=to_segment_sequence(data, True), time_delay=delay, n_states=n_states, cut=dict_args['cut'], data_with_repetion = True)
        if args.index_only:
            segments, repetition = segment_generator.calculate_recurrent_plot_points()
        else:
            segments, repetition = segment_generator.calculate_recurrent_segments()
        all_segments.append(segments)
        all_repetitions.append(repetition)
        
        if args.out_splitted_fragments:
            np.save(os.path.join(corpus_storage_base_path, f'{zone_type}_{data_index}_{sid}_d{delay}_s{n_states}.npy'), np.array(segments, dtype='object'), allow_pickle=True)
            np.save(os.path.join(corpus_storage_base_path, f'{zone_type}_{data_index}_{sid}_d{delay}_s{n_states}_repetition.npy'), np.array(repetition, dtype='object'), allow_pickle=True)

    if args.out_integrated_fragments:
        np.save(os.path.join(corpus_storage_base_path, 
                    f'{zone_type}_integrated_{sid}_d{delay}_s{n_states}.npy'), np.array(reduce(lambda seg1, seg2: seg1 + seg2, all_segments , []), dtype='object'), 
                    allow_pickle=True)
        np.save(os.path.join(corpus_storage_base_path, 
                    f'{zone_type}_integrated_{sid}_d{delay}_s{n_states}_repetition.npy'), np.array(reduce(lambda seg1, seg2: seg1 + seg2, all_repetitions , []), dtype='object'), 
                    allow_pickle=True)
        print(f"out {zone_type}_integrated_{sid}_d{delay}_s{n_states} Total {len(all_segments)} segments to {os.path.join(corpus_storage_base_path, f'{zone_type}_integrated_{sid}_d{delay}_s{n_states}.npy')}")
    

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

parser = argparse.ArgumentParser()
parser.add_argument("-cf", "--configuration-file", type=str, default="./configs/epilepsy_all_person_intergrated.json")
parser.add_argument("-i", "--index-only", type=bool, default=False)
parser.add_argument("-of", "--out_splitted_fragments", type=bool, default=False)
parser.add_argument("-om", "--out_integrated_fragments", type=bool, default=True)
parser.add_argument("-pz", "--pre_epileptic_zone", type = int, default=60 * 5) # area pz seconds before each seizure is identified as pre-epileptic area. 

args = parser.parse_args()

with open(args.configuration_file) as f:
    configuration_content = f.read()
    dict_args = json.loads(configuration_content)
    f.close()
    
dataset_base_path = dict_args.get('dataset_base_path', '.')
dataset_facade = EEGDatasetFacade(dataset_base_path=dataset_base_path)
dataset_name = dict_args.get("dataset_name", "")
dataset = dataset_facade(dataset_name)

corpus_storage_base_path = dict_args['corpus_storage_base_path']
microstate_storage_base_path = dict_args['microstate_storage_base_path']

print(f"dataset path = {dataset_base_path}")
print(f"dataset name = {dataset_name}")
print(f'length_of_pre_epileptic_zone = {args.pre_epileptic_zone} s')
print(f'microstate_storage_base_path = {microstate_storage_base_path}')
print(f'corpus_storage_base_path = {corpus_storage_base_path}')

if not os.path.exists(corpus_storage_base_path):
    os.makedirs(corpus_storage_base_path, exist_ok=True)
        
sids = dict_args['sids']     
delay = dict_args['delay']
n_states = dict_args['n_states']

for index, sid in enumerate(sids):
    print(f"processing record {sid}")
    
    # important: reduce_to_segments must be set to false.
    data = dataset.get_eeg_microstate_sequence(sid, dict_args['microstate_filename_form'], reduce_to_segments = False)
    data_total_length = data.shape[0] 

    print(f"data total length = {data_total_length}")
    sampling_frequency = dataset.get_sampling_rate()
    microstate_data = split_data(data, dict_args['merged_record_ids'][index], sampling_frequency)

    # check data_total_length == sum of segmentation lengths
    seizure_total_length = sum([len(segment) for segment in microstate_data['seizure']])
    normal_total_length = sum([len(segment) for segment in microstate_data['normal']])
    pre_epileptic_total_length = sum([len(segment) for segment in microstate_data['pre-epileptic']])
    data_total_length_after_splitting = seizure_total_length + normal_total_length + pre_epileptic_total_length

    if(data_total_length != data_total_length_after_splitting):
        print(f"data_total_length = {data_total_length}, data_total_length_after_splitting = {data_total_length_after_splitting}")
        assert False
    
    assert data_total_length == data_total_length_after_splitting, \
    f"Mismatch in data length: {data_total_length} != {data_total_length_after_splitting}"
    print("Check microstate length == data_total_length_after_splitting.. Passed")

    # build recurrence sentence by time-delay method.
    build_recurrence_sentences(microstate_data, "seizure")
    build_recurrence_sentences(microstate_data, "normal")
    build_recurrence_sentences(microstate_data, "pre-epileptic")