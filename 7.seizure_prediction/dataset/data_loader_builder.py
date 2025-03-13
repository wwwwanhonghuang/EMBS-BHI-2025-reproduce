from torch.utils.data import WeightedRandomSampler, DataLoader
from typing import List
import torch

def default_supervised_collate_fn():
    def collate_fn(batch):
        data = [item["data"] for item in batch]
        labels = torch.stack([item["labels"] for item in batch])

        return {"data": data, "values": labels}
    return collate_fn

def default_unsupervised_collate_fn():
    def collate_fn(batch):
        data = [item["data"] for item in batch]
        return {"data": data}
    return collate_fn

class DataLoaderBuilder:
    def __init__(self):
        pass

    @classmethod
    def build(cls, train_set, val_set, test_set, 
              train_sampler=None, val_sampler=None, test_sampler=None,
              shuffle_train_set=True, shuffle_val_set=True, shuffle_test_set=True,
              batch_size=32,
              collate_fn_train=None, collate_fn_val=None, collate_fn_test=None):

        if train_sampler:
            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                sampler=train_sampler,
                collate_fn=collate_fn_train
            )
        else:
            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=shuffle_train_set,
                collate_fn=collate_fn_train
            )
        
        if val_sampler:
            val_loader = DataLoader(
                val_set,
                batch_size=batch_size,
                sampler=val_sampler,
                collate_fn=collate_fn_val
            )
        else:
            val_loader = DataLoader(
                val_set,
                batch_size=batch_size,
                shuffle=shuffle_val_set,
                collate_fn=collate_fn_val
            )

        if test_sampler:
            test_loader = DataLoader(
                test_set,
                batch_size=batch_size,
                sampler=test_sampler,
                collate_fn=collate_fn_test
            )
        else:
            test_loader = DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=shuffle_test_set,
                collate_fn=collate_fn_test
            )

        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader
        }
