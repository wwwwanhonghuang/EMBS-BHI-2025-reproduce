
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

parser = argparse.ArgumentParser()
parser.add_argument("-cf", "--configuration-file", type=str, default="./configs/epilepsy_all_person_intergrated.json")
parser.add_argument("-i", "--index-only", type=bool, default=False)
parser.add_argument("-of", "--out_splitted_fragments", type=bool, default=False)
parser.add_argument("-om", "--out_integrated_fragments", type=bool, default=True)
parser.add_argument("-pz", "--pre_epileptic_zone", type = int, default=60 * 5) # area pz seconds before each seizure is identified as pre-epileptic area. 

args = parser.parse_args()

def split_data(data, record_ids, sampling_frequency):
    sampling_frequency = int(sampling_frequency)
    seizures = dataset.get_seizure_ranges_in_time_offsets(record_ids)
    length_of_pre_epileptic_zone = args.pre_epileptic_zone
    seizures = sorted(seizures, key=lambda x: x[0])
    print(seizures)
    splitted_data = {
        'seizures': [],
        'pre-epileptic':[],
        'normal': []
    }
    for seizure_annotation in seizures:
        splitted_data['seizures'].append(data[seizure_annotation[0]: seizure_annotation[1] + 1])
        print(f'added seizure segment {seizure_annotation[0]}:{seizure_annotation[1] + 1}')
    previous_seizure_ending_position = 0
    for seizure_annotation in seizures:
        normal_part_begin = previous_seizure_ending_position
        pre_epileptic_end = seizure_annotation[0]
        pre_epileptic_begin = max(seizure_annotation[0] - sampling_frequency * length_of_pre_epileptic_zone, previous_seizure_ending_position)
        normal_part_end = pre_epileptic_begin
        if(normal_part_end > normal_part_begin):
            splitted_data['normal'].append(data[normal_part_begin: normal_part_end])
            print(f'added normal segment {normal_part_begin}:{normal_part_end}, {data.shape}')

        if(pre_epileptic_end > pre_epileptic_begin):
            splitted_data['pre-epileptic'].append(data[pre_epileptic_begin: pre_epileptic_end])
            print(f'added pre-epileptic segment {pre_epileptic_begin}:{pre_epileptic_end}')

        previous_seizure_ending_position = seizure_annotation[1] + 1
    
    splitted_data['normal'].append(data[previous_seizure_ending_position: data.shape[0]])
    assert sum([len(segment) for segment in splitted_data['seizures']] + [len(segment) for segment in splitted_data['normal']] + [len(segment) for segment in splitted_data['pre-epileptic']]) == data.shape[0]
    print(f"seizures n_segments = {sum([len(segment) for segment in splitted_data['seizures']])}, normal n_segments = {sum([len(segment) for segment in splitted_data['normal']])}, pre-epileptic n_segments = {sum([len(segment) for segment in splitted_data['pre-epileptic']])}")
    print("Checks data length ==  total length of splitted fragments...Passed.")

    return splitted_data

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

for index, sid in enumerate(sids):
    print(f"processing record {sid}")
    delay = dict_args['delay']
    n_states = dict_args['n_states']

    data = dataset.get_eeg_microstate_sequence(sid, dict_args['microstate_filename_form'], reduce_to_segments = False)
    data_total_length = data.shape[0] 
    print(f"data total length = {data_total_length}")
    
    sampling_frequency = dataset.get_sampling_rate()
    data = split_data(data, dict_args['merged_record_ids'][index], sampling_frequency)
    
    data_total_length_after_splitting = sum([len(segment) for segment in data['seizures']]) + sum([len(segment) for segment in data['normal']]) +  sum([len(segment) for segment in data['pre-epileptic']])
    if(data_total_length != data_total_length_after_splitting):
        print(f"data_total_length = {data_total_length}, data_total_length_after_splitting = {data_total_length_after_splitting}")
        assert False
        
    assert data_total_length == data_total_length_after_splitting
    print("Check microstate length == data_total_length_after_splitting.. Passed")


    print("Proceeding seizure fragments...")
    all_segments = []
    for seizure_data_index, seizure_data in enumerate(data['seizures']):
        
        segment_generator = FiniteTimeDelaySegmentGenerator(data=to_segment_sequence(seizure_data), time_delay=delay, n_states=n_states, cut=dict_args['cut'])
        if args.index_only:
            segments = segment_generator.calculate_recurrent_plot_points()
        else:
            segments = segment_generator.calculate_recurrent_segments()
        all_segments.append(segments)
        if args.out_splitted_fragments:
            np.save(os.path.join(corpus_storage_base_path, f'seizure_{seizure_data_index}_{sid}_d{delay}_s{n_states}.npy'), np.array(segments, dtype='object'), allow_pickle=True)
    if args.out_integrated_fragments:
        np.save(os.path.join(corpus_storage_base_path, 
                    f'seizure_integrated_{sid}_d{delay}_s{n_states}.npy'), np.array(reduce(lambda seg1, seg2: seg1 + seg2, all_segments , []), dtype='object'), 
                    allow_pickle=True)
        print(f"out seizure_integrated_{sid}_d{delay}_s{n_states} Total {len(all_segments)} segments to {os.path.join(corpus_storage_base_path, f'seizure_integrated_{sid}_d{delay}_s{n_states}.npy')}")
    
    print("Proceeding normal fragments...")
    all_segments = []
    for normal_data_index, normal_data in enumerate(data['normal']):
        segment_generator = FiniteTimeDelaySegmentGenerator(data=to_segment_sequence(normal_data), time_delay=delay, n_states=n_states, cut=dict_args['cut'])
        if args.index_only:
            segments = segment_generator.calculate_recurrent_plot_points()
        else:
            segments = segment_generator.calculate_recurrent_segments()
        all_segments.append(segments)
        if args.out_splitted_fragments:
            np.save(os.path.join(corpus_storage_base_path, f'normal_{seizure_data_index}_{sid}_d{delay}_s{n_states}.npy'), np.array(segments, dtype='object'), allow_pickle=True)
    if args.out_integrated_fragments:
        np.save(os.path.join(corpus_storage_base_path, 
            f'normal_integrated_{sid}_d{delay}_s{n_states}.npy'), np.array(reduce(lambda seg1, seg2: seg1 + seg2, all_segments , []), dtype='object'), 
                    allow_pickle=True)
    
    print("Proceeding pre-epileptic fragments...")

    all_segments = []
    for pre_epileptic_data_index, pre_epileptic_data in enumerate(data['pre-epileptic']):
        segment_generator = FiniteTimeDelaySegmentGenerator(data=to_segment_sequence(pre_epileptic_data), time_delay=delay, n_states=n_states, cut=dict_args['cut'])
        if args.index_only:
            segments = segment_generator.calculate_recurrent_plot_points()
        else:
            segments = segment_generator.calculate_recurrent_segments()
        all_segments.append(segments)
        if args.out_splitted_fragments:
            np.save(os.path.join(corpus_storage_base_path, 
                f'pre-epileptic_{seizure_data_index}_{sid}_d{delay}_s{n_states}.npy'), 
                np.array(segments, dtype='object'), allow_pickle=True)
    
    if args.out_integrated_fragments:
        np.save(os.path.join(corpus_storage_base_path, 
                    f'pre-epileptic_integrated_{sid}_d{delay}_s{n_states}.npy'), np.array(reduce(lambda seg1, seg2: seg1 + seg2, all_segments , []), dtype='object'), 
                    allow_pickle=True)

