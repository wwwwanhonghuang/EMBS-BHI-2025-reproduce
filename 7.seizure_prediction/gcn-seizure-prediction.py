import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, WeightedRandomSampler
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch.nn as nn
from functools import lru_cache

tree_records_base_path = "../data/serialized_tree"

class SyntaxTreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value  # Tuple of 6 values
        self.left = left
        self.right = right
    @property
    def children(self):
        return [self.left, self.right]
        
    def __repr__(self):
        return f"SyntaxTreeNode(value={self.value}, left={self.left}, right={self.right})"


def deserialize_tree(tree_str):
    tokens = tree_str.split()
    index = 0

    def _deserialize_helper():
        nonlocal index
        if index >= len(tokens):
            return None

        if tokens[index] == "#":
            index += 1
            return None

        # Extract the 6 values for the current node
        value = (
            int(tokens[index]),     # std::get<0>
            int(tokens[index + 1]), # std::get<1>
            int(tokens[index + 2]), # std::get<2>
            int(tokens[index + 3]), # std::get<3>
            float(tokens[index + 4]), # std::get<4>
            int(tokens[index + 5])  # std::get<5>
        )
        index += 6

        # Recursively deserialize left and right children
        left = _deserialize_helper()
        right = _deserialize_helper()

        return SyntaxTreeNode(value, left, right)

    return _deserialize_helper()


def tree_to_graph(root):
    """
    Convert a SyntaxTreeNode to a PyTorch Geometric graph.
    """
    
    nodes = []
    edges = []
    node_id_map = {}

    def dfs(node, parent_id=None):
        if (node is None):
            return
            
        # Assign a unique ID to the node
        node_id = len(nodes)
        node_id_map[node] = node_id
        nodes.append(node)

        # Add edge from parent to current node
        if parent_id is not None:
            edges.append((parent_id, node_id))

        # Recursively process children
        for child in node.children:
            dfs(child, node_id)

    dfs(root)
    
    x = []
    for node in nodes:
        # Assume node.features is an order-6 tuple
        features = node.value

        # Separate dimensions
        feature_0 = (int)(features[0]) + 1
        feature_0 = 0 if feature_0 == 65536 else feature_0
        
        feature_1 = (int)(features[1]) + 1
        feature_1 = 0 if feature_1 == 65536 else feature_1
        
        feature_2 = (int)(features[2]) + 1
        feature_2 = 0 if feature_2 == 65536 else feature_2

        feature_5 = (int)(features[5]) + 1
        feature_5 = 0 if feature_5 == 65536 else feature_5
   
        possibility = features[4]
        
        
        # Concatenate all features into a single vector
        node_features = torch.tensor([possibility, feature_0, feature_1, feature_2, feature_5], dtype = torch.float)
        x.append(node_features)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # Edge indices

    return Data(
        x=torch.stack(x),  # Node feature matrix
        edge_index=edge_index # Edge index
    )
    
# Example serialized tree string
tree_str = "1 2 3 4 5.0 6 7 8 9 10 11.0 12 # # #"

# Deserialize the tree
root = deserialize_tree(tree_str)
print(root)
# Convert the tree to a graph
graph = tree_to_graph(root)

# Print the graph
print(graph)
print("Node features:", graph.x)
print("Edge index:", graph.edge_index)

from torch_geometric.data import Batch

class TreeDataset(Dataset):
    
    def __init__(self, area_types):
        super(TreeDataset, self).__init__()
        files = []
        labels = []
        self.label_map = {
            'normal-retained': 0,
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
        graph = tree_to_graph(root)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {"graph": graph, "label":label}


dataset_types = ["normal-retained", "seizure", "pre-epileptic"]

dataset = TreeDataset(dataset_types)

from sklearn.model_selection import train_test_split
import torch
# Assuming 'dataset' is an instance of TreeDataset
train_idx, temp_idx = train_test_split(
    list(range(len(dataset))), 
    test_size=0.2, 
    stratify=dataset.labels
)

val_idx, test_idx = train_test_split(
    temp_idx, 
    test_size=0.5, 
    stratify=[dataset.labels[i] for i in temp_idx]  # This is fine since temp_idx contains the original indices
)

# Create subsets
train_subset = torch.utils.data.Subset(dataset, train_idx)
val_subset = torch.utils.data.Subset(dataset, val_idx)
test_subset = torch.utils.data.Subset(dataset, test_idx)


# Now we calculate the weights based on the train_subset (not the full dataset)
train_labels = [dataset.labels[i] for i in train_idx]

# Calculate class counts for the training set
class_counts = [0, 0, 0]  # For normal, seizure, pre-epileptic
for label in train_labels:
    class_counts[label] += 1

total_samples = len(train_labels)
class_weights = [total_samples / count for count in class_counts]

# Calculate sample weights for the training set
sample_weights = [class_weights[label] for label in train_labels]

# Create the sampler for the training set
train_sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)


# Print sizes to confirm
print(f"Train size: {len(train_subset)}")
print(f"Validation size: {len(val_subset)}")
print(f"Test size: {len(test_subset)}")

def custom_collate(batch):
    """
    Custom collate function to batch torch_geometric.data.Data objects.
    Args:
        batch: List of dictionaries containing "graph" and "label".
    Returns:
        Batched graphs and labels.
    """
    graphs = [item["graph"] for item in batch]
    labels = [item["label"] for item in batch]
    
    # Batch graphs using PyTorch Geometric's Batch class
    batched_graphs = Batch.from_data_list(graphs)
    
    # Stack labels into a tensor
    batched_labels = torch.stack(labels)
    
    return batched_graphs, batched_labels
    
# Create DataLoader for the training set with WeightedRandomSampler
train_loader = DataLoader(
    train_subset,
    batch_size=32,
    sampler=train_sampler,
    collate_fn=custom_collate  # Use custom collate function
)

# Create DataLoader for the validation and test sets without any sampling (just shuffle them)
val_loader = DataLoader(
    val_subset,
    batch_size=32,
    shuffle=True,
    collate_fn=custom_collate  # Use custom collate function
)

test_loader = DataLoader(
    test_subset,
    batch_size=32,
    shuffle=True,
    collate_fn=custom_collate  # Use custom collate function
)

from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool



class TreeGNN(torch.nn.Module):
    def __init__(self, hidden_dim=128, num_classes = 3):
        super(TreeGNN, self).__init__()
        # Define embedding layers
        self.dim0_embedding = nn.Embedding(num_embeddings=96, embedding_dim=32)
        self.dim1_embedding = nn.Embedding(num_embeddings=96, embedding_dim=32)
        self.dim2_embedding = nn.Embedding(num_embeddings=96, embedding_dim=32)
        self.dim5_embedding = nn.Embedding(num_embeddings=182, embedding_dim=32)

        # Define GNN layers
        self.conv1 = GCNConv(129, hidden_dim)  # Input size: 1 + 32*4 = 129
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)


    def forward(self, x, edge_index, batch):
        # Separate features
        possibility = x[:, 0].unsqueeze(1)  # Shape: [num_nodes, 1]
        dim0 = x[:, 1].long()
        dim1 = x[:, 2].long()
        dim2 = x[:, 3].long()
        dim5 = x[:, 4].long()

        # Embed categorical features
        dim0_embedded = self.dim0_embedding(dim0)  # Shape: [num_nodes, 16]
        dim1_embedded = self.dim1_embedding(dim1)  # Shape: [num_nodes, 16]
        dim2_embedded = self.dim2_embedding(dim2)  # Shape: [num_nodes, 16]
        dim5_embedded = self.dim5_embedding(dim5)  # Shape: [num_nodes, 16]

        # Concatenate all features
        x = torch.cat([possibility, dim0_embedded, dim1_embedded, dim2_embedded, dim5_embedded], dim=1)
        # Pass through GNN layers
        x = F.relu(self.bn1(self.conv1(x, edge_index)))  # Shape: [num_nodes, hidden_dim]
        x = F.relu(self.bn2(self.conv2(x, edge_index)))          # Shape: [num_nodes, output_dim]
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)  # Aggregate node features into graph-level features

        x = self.fc(x)
        return x


model = TreeGNN(hidden_dim=256,  num_classes = 3)
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np

class_weights = torch.tensor([1.0, 1.0, 1.0])  # Higher weight for class 1 (seizure)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Cross-Entropy Loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.0001)
def calculate_metrics(conf_matrix):
    """
    Calculate accuracy, TPR, FPR, TNR, FNR, F1, and F2 from a confusion matrix.
    Args:
        conf_matrix: Confusion matrix (3x3 for 3 classes).
    Returns:
        Dictionary of metrics.
    """
    metrics = {}
    
    # True Positives (diagonal of the confusion matrix)
    TP = np.diag(conf_matrix)
    
    # False Positives (sum of columns minus diagonal)
    FP = np.sum(conf_matrix, axis=0) - TP
    
    # False Negatives (sum of rows minus diagonal)
    FN = np.sum(conf_matrix, axis=1) - TP
    
    # True Negatives (total samples minus TP, FP, FN)
    TN = np.sum(conf_matrix) - (TP + FP + FN)
    
    # Accuracy
    metrics["accuracy"] = np.sum(TP) / np.sum(conf_matrix)
    
    # True Positive Rate (Recall)
    metrics["TPR"] = np.divide(TP, TP + FN, where=(TP + FN) != 0)
    
    # False Positive Rate
    metrics["FPR"] = np.divide(FP, FP + TN, where=(FP + TN) != 0)
    
    # True Negative Rate
    metrics["TNR"] = np.divide(TN, TN + FP, where=(TN + FP) != 0)
    
    # False Negative Rate
    metrics["FNR"] = np.divide(FN, TP + FN, where=(TP + FN) != 0)
    
    # Precision
    precision = np.divide(TP, TP + FP, where=(TP + FP) != 0)
    
    # F1 Score
    metrics["F1"] = np.divide(2 * (precision * metrics["TPR"]), (precision + metrics["TPR"]), where=(precision + metrics["TPR"]) != 0)
    
    # F2 Score
    metrics["F2"] = np.divide(5 * (precision * metrics["TPR"]), (4 * precision + metrics["TPR"]), where=(4 * precision + metrics["TPR"]) != 0)
    
    return metrics

# Training loop
for epoch in range(100):
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
    
