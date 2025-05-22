
from torch.utils.data import Dataset
import torch
import os
from functools import lru_cache
from utils.tree_helper import deserialize_tree, tree_to_graph


class GCNTreeDataset(Dataset):
    
    def __init__(self, tree_records_base_path, area_types, device = 'cpu'):
        super(GCNTreeDataset, self).__init__()
        files = []
        labels = []
        self.label_map = {
            'normal': 0,
            'seizure': 1,
            'pre-epileptic': 2
        }
        
        for area_type in area_types:
            dataset_base_path = os.path.join(tree_records_base_path, area_type)
            files_this_area_type = ([os.path.join(dataset_base_path, file) for file in os.listdir(dataset_base_path)])
            files += files_this_area_type
            labels += [self.label_map[area_type]] * len(files_this_area_type)
            print(f'Add {len(files_this_area_type)} for category {area_type}')
            
        self.files = files
        self.labels = labels
        assert(len(self.files) == len(self.labels))
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if(idx >= len(self.files)):
            raise ValueError(f'idx = {idx} >= total amount of files = {len(self.files)}')
        file = self.files[int(idx)]
        with open(file, "r") as f:
            serialized_tree = f.read().strip()
        root = deserialize_tree(serialized_tree)
        graph = tree_to_graph(root, self.device)
        label = torch.tensor(self.labels[int(idx)], dtype=torch.long).to(self.device)
        return {"graph": graph, "label":label}


class HighOrderGCNTreeDataset(Dataset):
    
    def __init__(self, tree_records_base_path, area_types, order=2, device='cpu'):
        super(HighOrderGCNTreeDataset, self).__init__()
        files = []
        labels = []
        self.label_map = {
            'normal': 0,
            'seizure': 1,
            'pre-epileptic': 2
        }
        
        # Collecting files and corresponding labels
        for area_type in area_types:
            dataset_base_path = os.path.join(tree_records_base_path, area_type)
            files_this_area_type = [os.path.join(dataset_base_path, file) for file in os.listdir(dataset_base_path)]
            files += files_this_area_type
            labels += [self.label_map[area_type]] * len(files_this_area_type)
            print(f'Add {len(files_this_area_type)} for category {area_type}')
        
        # Do not shuffle the files list as file order is significant (temporal order)
        self.files = files
        self.labels = labels
        assert(len(self.files) == len(self.labels))  # Check that files and labels match
        self.device = device
        self.order = order

    def __len__(self):
        return len(self.files) - self.order + 1

    @lru_cache
    def __getitem__(self, idx):
        # Ensure idx is within valid range
        if idx >= len(self.files) + self.order - 1:
            raise ValueError(f'idx = {idx} is out of range. Valid range: {0} to {len(self.files) + self.order - 1}')
        
        # Get the sequence of files based on the order
        files = self.files[int(idx): int(idx) + self.order - 1]
        
        # Deserialize and convert each file to graph
        serialized_trees = []
        for file in files:
            with open(file, "r") as f:
                serialized_tree = f.read().strip()
            serialized_trees.append(serialized_tree)
            
        trees = []
        for serialized_tree in serialized_trees:
            root = deserialize_tree(serialized_tree)
            graph = tree_to_graph(root, self.device)
            tree_data = {
                'root': root,
                'graph': graph,
            }
            trees.append(tree_data)
        
        # Use the label of the final tree in the sequence
        label = torch.tensor(self.labels[int(idx) + self.order - 1], dtype=torch.long).to(self.device)

        # Return a dictionary with graphs and their corresponding label
        return {"graphs": [tree_data['graph'] for tree_data in trees], "label": label}


class TreeLSTMTreeDataset(Dataset):    
    def __init__(self, area_types, tree_records_base_path):
        super(TreeLSTMTreeDataset, self).__init__()
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
