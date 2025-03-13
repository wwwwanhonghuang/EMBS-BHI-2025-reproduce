from sklearn.model_selection import train_test_split
import torch

class RatioBasedDataSetSplitter():
    def __init__(self, dataset):
        self.dataset = dataset
        
    def split_dataset(self, train_set_percentage = 0.8, 
                      test_set_percentage = 0.1, validate_set_percentage = 0.1, return_indexes = True):
        dataset = self.dataset
        assert test_set_percentage + validate_set_percentage + train_set_percentage == 1.0, "Error: test_set_percentage + validate_set_percentage + train_set_percentage != 1.0"

        train_idx, temp_idx = train_test_split(
            list(range(len(dataset))), 
            test_size=test_set_percentage + validate_set_percentage, 
            stratify=dataset.labels
        )

        val_idx, test_idx = train_test_split(
            temp_idx, 
            test_size=test_set_percentage / (test_set_percentage + validate_set_percentage), 
            stratify=[dataset.labels[i] for i in temp_idx]
        )

        # Create subsets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        test_subset = torch.utils.data.Subset(dataset, test_idx)

        # train_labels = [dataset.labels[i] for i in train_idx]

        # # Calculate class counts for the training set
        # class_counts = [0, 0, 0]  # For normal, seizure, pre-epileptic
        # for label in train_labels:
        #     class_counts[label] += 1

        # total_samples = len(train_labels)
        # class_weights = [total_samples / count for count in class_counts]

        # # Calculate sample weights for the training set
        # sample_weights = [class_weights[label] for label in train_labels]

        # # Create the sampler for the training set
        # train_sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)


        # Print sizes to confirm
        print(f"Train size: {len(train_subset)}")
        print(f"Validation size: {len(val_subset)}")
        print(f"Test size: {len(test_subset)}")
        
        sample_indexes = {

        } if not return_indexes else {
            "train_set_indexes": train_idx,
            "val_set_indexes": val_idx,
            "test_set_indexes": test_idx
        }

        return {
            "train_set": train_subset,
            "val_set": val_subset,
            "test_set": test_idx
        } | sample_indexes
