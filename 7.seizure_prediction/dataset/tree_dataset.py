
from torch.utils.data import Dataset
import torch
import os
from functools import lru_cache
from utils.tree_helper import deserialize_tree

class TreeDataset(Dataset):    
    def __init__(self, area_types, tree_records_base_path):
        super(TreeDataset, self).__init__()
        files = []
        labels = []
        self.label_map = {
            'normal-retained': 0,
            'seizure': 1,
            'pre-epileptic': 2
        }
        self.tree_records_base_path = tree_records_base_path
        
        for area_type in area_types:
            dataset_base_path = os.path.join(tree_records_base_path, area_type)
            files_this_area_type = ([os.path.join(dataset_base_path, file) for file in os.listdir(dataset_base_path)])
            files += files_this_area_type
            labels += [self.label_map[area_type]] * len(files_this_area_type)
            print(f'Add {len(files_this_area_type)} for category {area_type}')
            
        self.files = files
        self.labels = labels
        assert(len(self.files) == len(self.labels))

    def __len__(self):
        return len(self.files)

    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        if(idx >= len(self.files)):
            raise ValueError(f'idx = {idx} >= total amount of files = {len(self.files)}')
        file = self.files[idx]
        with open(file, "r") as f:
            serialized_tree = f.read().strip()
        root = deserialize_tree(serialized_tree)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {"tree": root.to_dict(), "labels":label}