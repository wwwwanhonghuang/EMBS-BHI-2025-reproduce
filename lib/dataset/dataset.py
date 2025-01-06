import numpy as np
import mne
import os, sys
import json
import importlib.util   
from dataset.preprocessing import *

class BaseEEGDataset():
    def __init__(self, dataset_base_path, dataset_info, dataset_name):
        self.base_path = dataset_base_path
        self.dataset_name = dataset_name
        self.dataset_info = dataset_info

    def get_seizure_annotations(self, record_ids):
        raise NotImplementedError
        
    def get_preprocessing_pipeline(self, key):
        pipeline_config = self.preprocessings[key]
        return PreprocessingPipeline(pipeline_config)
    
    def get_numpy_data(self, index):
        return self.get_mne_data(index).get_data()
    
    def get_multiple_numpy_data(self, indexes):
        return [self.get_numpy_data(index) for index in indexes]
    
    def get_merge_numpy_data(self, indexes):
        data = self.get_multiple_numpy_data(indexes)
        return np.concatenate(data, axis = 1)
        
    def get_mne_data(self, index) -> mne.io.BaseRaw:
        raise NotImplemented
    
    def get_multiple_mne_data(self, indexes):
        return [self.get_mne_data(index) for index in indexes]
    
    def get_merge_mne_data(self, indexes):
        data = self.get_multiple_mne_data(indexes)
        return mne.concatenate_raws(data)

class DatasetController():
    def get_dataset(self, dataset_name, class_name, dataset_base_path, dataset_info):
        def import_dataset_class(dataset_base_path, class_name):
            module_path = os.path.join(dataset_base_path, "dataset.py")
            spec = importlib.util.spec_from_file_location(dataset_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules['module_name'] = module
            return getattr(module, class_name)
            
        def instantiatize_dataset_class():
            return import_dataset_class(dataset_base_path, class_name)(dataset_base_path, dataset_info)
        
        if "epileptic_eeg_dataset" == dataset_name:
            return instantiatize_dataset_class()
        if "ethz-ieeg" == dataset_name:
            return instantiatize_dataset_class()
        
class EEGDatasetFacade():
    def __init__(self, dataset_base_path = ".", dataset_info_file_name = "dataset_info.json") -> None:
        self.dataset_base_path = dataset_base_path
        dataset_info_file_path = os.path.join(dataset_base_path, dataset_info_file_name)
        with open(dataset_info_file_path) as f:
            self.dataset_info = json.loads(f.read())
        self.controller = DatasetController()
        
    def __call__(self, dataset_name):
        if self.dataset_info is None or dataset_name not in self.dataset_info:
            return None
        return self.controller.get_dataset(dataset_name, \
            self.dataset_info[dataset_name]['class_name'], \
            os.path.join(self.dataset_base_path, self.dataset_info[dataset_name]['path']), \
            dataset_info = self.dataset_info[dataset_name])
    
    
        

class MicrostateEnabledEEGDataset(BaseEEGDataset):
    def __init__(self, dataset_base_path, preprocessings, dataset_name):
        super(MicrostateEnabledEEGDataset, self).__init__(dataset_base_path, preprocessings, dataset_name)
