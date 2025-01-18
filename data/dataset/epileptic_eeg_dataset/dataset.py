import os, sys
sys.path.append("..")
sys.path.append("../lib/dataset")

from dataset import BaseEEGDataset, MicrostateEnabledEEGDataset
from experiment_utils import to_segment_sequence

import mne
import numpy as np

from data.dataset.epileptic_eeg_dataset.annotations import seizure_times, onset_times

class EpilepticEEGDataset(MicrostateEnabledEEGDataset):
    def __init__(self, dataset_base_path, dataset_info):
        super().__init__(dataset_base_path, dataset_info, "epileptic_eeg_dataset")
        self.seizure_times = seizure_times
        self.onset_times = onset_times
        self.sampling_rate = 500.0
        
    def get_sampling_rate(self):
        return self.sampling_rate
    
    def get_mne_data(self, index):
        person_id = index[0]
        record_id = index[1]
        filepath = os.path.join(self.base_path, "Raw_EDF_Files", f'p{person_id}_Record{record_id}.edf')
        print(filepath)
        return mne.io.read_raw(filepath)
    
    def get_eeg_microstate_sequence(self, sid, microstate_filename_form, reduce_to_segments = True):
        import re
        loaded = False
        microstate_sequence_filename_pattern = microstate_filename_form.replace("#{sid}", str(sid))

        microstate_storage_base_path = self.dataset_info['microstate_storage_base_path']
        for file_name in os.listdir(microstate_storage_base_path):
            
            match = re.match(microstate_sequence_filename_pattern, file_name)
            if match is not None:
                print(f"Processing file {file_name}.. Load {os.path.join(microstate_storage_base_path, file_name)}")
                data = np.load(os.path.join(microstate_storage_base_path, file_name))
                if reduce_to_segments:
                    data = to_segment_sequence(data)
                loaded = True
        if not loaded:
            raise FileNotFoundError(f"File for study_id = {sid} not found.")
        return data
    
    def get_seizure_ranges_in_time_offsets(self, record_ids):
        """
        Retrieve seizure intervals in global timestamps for aggregated EEG data.

        Args:
            record_ids (set): A set of record IDs.

        Returns:
            list: A list of seizure intervals in global timestamps corresponding 
                to the aggregated EEG data of the specified records.
        """
        seizures = []
        prefix_total_n_time = 0

        HOUR_INDEX = 0
        MINUTE_INDEX = 1
        SECOND_INDEX = 2
        DURATION_INDEX = 3

        # For each record
        for index in record_ids:
            print(index)
            # read the eeg data, frequency, times
            mne_data = self.get_mne_data(index)
            freq = int(mne_data.info['sfreq'])
            length = int(mne_data.n_times)
            # read the seizure annotation (in local time stamp of this record)
            person_record_seizures = self.seizure_times[index[0]][index[1]]
            record_onset_time = onset_times[index[0]][index[1]]
            # convert annotation to global timestamp intervals.
            for seizure in person_record_seizures:
                time_offset = np.array(seizure[:3]) - np.array(record_onset_time)
                duration = seizure[DURATION_INDEX]
                time_diff_to_recording_onset_in_secs = (time_offset[HOUR_INDEX] * 3600 + 
                    time_offset[MINUTE_INDEX] * 60 + time_offset[SECOND_INDEX])
                
                start_n_time_offset = time_diff_to_recording_onset_in_secs * freq
                end_n_time_offset = start_n_time_offset + duration * freq
                print(f"seizure of person_{index[0]}, record_{index[1]} from time step {start_n_time_offset} to {end_n_time_offset}")
                
                # check the interval is legal.
                assert start_n_time_offset >= 0, f"Negative start time offset for seizure {seizure} in record {index}"
                assert end_n_time_offset >= 0, f"Negative end time offset for seizure {seizure} in record {index}"
                assert end_n_time_offset >= start_n_time_offset, f"interval end {end_n_time_offset} should >= interval start {start_n_time_offset}"
                seizures.append((prefix_total_n_time + start_n_time_offset, prefix_total_n_time + end_n_time_offset))
            prefix_total_n_time += length
        return seizures
    
    def get_seizure_annotations(self, record_ids):
        raise ValueError
    
