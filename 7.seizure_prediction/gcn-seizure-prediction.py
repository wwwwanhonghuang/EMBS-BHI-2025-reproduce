import numpy as np
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch_geometric.data import Dataset, Batch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from utils.utils import calculate_metrics

from models.tree_gnn import TreeGNN


from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np
import torch
from models.focal_loss import FocalLoss
from dataset.tree_dataset import GCNTreeDataset


def split_class_indices(class_indices, test_size=0.2, val_size=0.1):
    """Split a single class's indices into train, val, test."""
    # First split: train (80%) vs temp (20%)
    train_idx, temp_idx = train_test_split(
        class_indices,
        test_size=test_size
    )
    # Second split: val (50% of temp) vs test (50% of temp)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5
    )
    return train_idx, val_idx, test_idx

parser = argparse.ArgumentParser()
parser.add_argument("--tree_records_base_path", type=str, default="/data1/pcfg-log/serialized_tree")
parser.add_argument("--save_folder", type=str, default="./save")
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--binary", dest='binary', action='store_true')
parser.set_defaults(binary=False)

args = parser.parse_args()

tree_records_base_path = args.tree_records_base_path
save_folder = args.save_folder

if not os.path.exists(save_folder):
    os.makedirs(save_folder)


dataset_types = ["normal", "seizure", "pre-epileptic"]
    
device = 'cpu'
dataset = GCNTreeDataset(tree_records_base_path, dataset_types, device=device)

## Class Sample Filtering
# First, get all indices grouped by class
class_indices = defaultdict(list)
for idx, label in enumerate(dataset.labels):
    class_indices[label].append(idx)

num_class0_keep = len(class_indices[0])  
class0_indices = np.random.choice(class_indices[0], size=num_class0_keep, replace=False) # normal area

# Keep all of class 1 and class 2
class1_indices = class_indices[1] # seizure area

if not args.binary:
    class2_indices = class_indices[2]
else:
    class2_indices = []
    class0_indices += class_indices[2] # treat preepileptic area as normal

# Split each class individually
class0_train, class0_val, class0_test = split_class_indices(class0_indices)
class1_train, class1_val, class1_test = split_class_indices(class1_indices)

if not args.binary:
    class2_train, class2_val, class2_test = split_class_indices(class2_indices)

cnt_class_0 = {
    'type': 'cnt_class_0',
    'train': len(class0_train),
    'val': len(class0_val),
    'test': len(class0_test)
}

cnt_class_1 = {
    'type': 'cnt_class_1',
    'train': len(class1_train),
    'val': len(class1_val),
    'test': len(class1_test)
}

if not args.binary:
    cnt_class_2 = {
        'type': 'cnt_class_2',
        'train': len(class2_train),
        'val': len(class2_val),
        'test': len(class2_test)
    }
    
if not args.binary:
    print(cnt_class_0, cnt_class_1, cnt_class_2)
else:
    print(cnt_class_0, cnt_class_1)

# Merge splits across classes
train_idx = np.concatenate([class0_train, class1_train, class2_train])
val_idx = np.concatenate([class0_val, class1_val, class2_val])
test_idx = np.concatenate([class0_test, class1_test, class2_test])

if(len(set(dataset.labels[i] for i in train_idx))) < 3:
    print("Warning: train set contains classes less than 3.")
if(len(set(dataset.labels[i] for i in val_idx))) < 3:
    print("Warning: validation set contains classes less than 3.")
if(len(set(dataset.labels[i] for i in val_idx))) < 3:
    print("Warning: test set contains classes less than 3.")

# Create subsets
train_subset = torch.utils.data.Subset(dataset, train_idx)
val_subset = torch.utils.data.Subset(dataset, val_idx)
test_subset = torch.utils.data.Subset(dataset, test_idx)

if not args.binary:
    train_labels = [dataset.labels[i] for i in train_idx]
    if(len(set(train_labels))) < 3:
        print("Warning: test set contains classes less than 3.")
else:
    train_labels = [1 if dataset.labels[i] == 1 else 0 for i in train_idx]

# Calculate class counts for the training set
if not args.binary:
    class_counts = [len(class0_indices), len(class1_indices), len(class2_indices)]  # For normal, seizure, pre-epileptic
else:
    class_counts = [len(class0_indices), len(class1_indices)]  # For normal + pre-epileptic, seizure

print(f'class counts: {class_counts}')

# total_samples = len(train_labels)
# class_weights = [total_samples / count for count in class_counts]

# beta = 0.9  
# effective_num = [1.0 - np.exp(-beta * np.log(count + 1)) for count in class_counts]
# class_weights = [(1.0 - beta) / en for en in effective_num]

class_weights = 1.0 / (np.array(class_counts) + 1e-5)  # Add epsilon to avoid division by zero
class_weights = class_weights / np.sum(class_weights)  # Normalize (optional)

# Calculate sample weights for the training set
sample_weights = [class_weights[label] for label in train_labels]
print(f"class weights = {class_weights}")

# Create the sampler for the training set
train_sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)

# Print sizes to confirm
print(f"Train size: {len(train_subset)}")
print(f"Validation size: {len(val_subset)}")
print(f"Test size: {len(test_subset)}")

binary_classification=args.binary

def custom_collate(batch):
    """
    Custom collate function to batch torch_geometric.data.Data objects.
    Args:
        batch: List of dictionaries containing "graph" and "label".
    Returns:
        Batched graphs and labels.
    """
    graphs = [item["graph"] for item in batch]
    if not binary_classification:
        labels = [item["label"] for item in batch]
    else:
        labels = [1 if item["label"] == 1 else 0 for item in batch]

    
    # Batch graphs using PyTorch Geometric's Batch class
    batched_graphs = Batch.from_data_list(graphs).to(device)
    
    # Stack labels into a tensor
    batched_labels = torch.stack(labels).to(device)
    
    return batched_graphs, batched_labels
    
# Create DataLoader for the training set with WeightedRandomSampler
train_loader = DataLoader(
    train_subset,
    batch_size=64,
    sampler=train_sampler,
    collate_fn=custom_collate  # Use custom collate function
)

# Create DataLoader for the validation and test sets without any sampling (just shuffle them)
val_loader = DataLoader(
    val_subset,
    batch_size=64,
    shuffle=True,
    collate_fn=custom_collate  # Use custom collate function
)

test_loader = DataLoader(
    test_subset,
    batch_size=64,
    shuffle=True,
    collate_fn=custom_collate  # Use custom collate function
)

if not args.binary:
    num_classes = 3
else:
    num_classes = 2

model = TreeGNN(hidden_dim = 256,  num_classes = num_classes).to(device)

if args.binary:
    criterion = torch.nn.BCEWithLogitsLoss()
else:
    # class_weights = torch.tensor([1.0, 1.0, 1.0]).to(device)
    # criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
#    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
    criterion = FocalLoss(alpha=[1.0, 5.0, 5.0], gamma=2.0)

# Cross-Entropy Loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

metrics_dict = {}
n_epochs = args.n_epochs

# Training loop
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        # Forward pass: get predictions
        graphs, labels = batch['graph'], batch['label']
        output = model(graphs.x, graphs.edge_index, graphs.batch)
        # Compute the loss
        loss = criterion(output, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Store predictions and labels for metrics
        preds = torch.argmax(output, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculate training metrics
    train_conf_matrix = confusion_matrix(all_labels, all_preds)
    train_metrics = calculate_metrics(train_conf_matrix)
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
    print(f"Training Metrics: {train_metrics}")

    # Validation
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            graphs, labels = batch['graph'], batch['label']
            output = model(graphs.x, graphs.edge_index, graphs.batch)
            loss = criterion(output, labels)
            val_loss += loss.item()
            
            # Store predictions and labels for metrics
            preds = torch.argmax(output, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    # Calculate validation metrics
    val_conf_matrix = confusion_matrix(val_labels, val_preds)
    val_metrics = calculate_metrics(val_conf_matrix)
    
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Metrics: {val_metrics}")
    
    torch.save(model.state_dict(), os.path.join(save_folder , f'gcn_model_epoch_{epoch}.pth'))
    
    
    model.eval()
    test_loss = 0
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            graphs, labels = batch['graph'], batch['label']
            output = model(graphs.x, graphs.edge_index, graphs.batch)
            loss = criterion(output, labels)
            test_loss += loss.item()
            
            # Store predictions and labels for metrics
            preds = torch.argmax(output, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    # Calculate Test metrics
    test_conf_matrix = confusion_matrix(test_labels, test_preds)
    test_metrics = calculate_metrics(test_conf_matrix)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Metrics: {test_metrics}")
    
    metrics_dict[epoch] = {
    'train_loss': total_loss,
    'train_metrics': train_metrics.tolist() if isinstance(train_metrics, np.ndarray) else train_metrics,
    'val_loss': val_loss,
    'val_metrics': val_metrics.tolist() if isinstance(val_metrics, np.ndarray) else val_metrics,
    'test_loss': test_loss,
    'test_metrics': test_metrics.tolist() if isinstance(test_metrics, np.ndarray) else test_metrics,
    }

    np.save(os.path.join(save_folder, f'metrics_epoch_{epoch}.npy'), metrics_dict, allow_pickle=True)
      
