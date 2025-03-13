from torch.utils.data import WeightedRandomSampler
from typing import List

class ProportionalWeightedRandomSamplerBuilder():
    def __init__(self):
        pass

    @classmethod
    def build(cls, n_classes, labels: List[int], return_weights = True):
        unique_labels = sorted(set(labels))         
        class_counts = {label: 0 for label in unique_labels} 
        
        for label in labels:
            class_counts[label] += 1

        total_samples = len(labels)

        class_weights = {label: total_samples / count if count > 0 else 0 for label, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in labels]

        train_sampler = WeightedRandomSampler(sample_weights, len(labels), replacement=True)

        if not return_weights:
            return train_sampler
        else:
            return train_sampler, class_weights