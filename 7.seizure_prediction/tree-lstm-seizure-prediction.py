from tqdm import tqdm
import torch
import torch.optim as optim
from torch import nn

from data_structures.tree import SyntaxTreeNode
from dataset.tree_dataset import TreeDataset

from dataset.samplers import ProportionalWeightedRandomSamplerBuilder
from dataset.dataset_splitter import RatioBasedDataSetSplitter
from dataset.data_loader_builder import default_supervised_collate_fn, DataLoaderBuilder

from models.tree_lstm import SeizurePredictionInputEmbeddingPreprocessor, BinaryTreeLSTMCell, BinaryTreeLSTM
from utils.utils import calculate_metrics

from sklearn.metrics import confusion_matrix
import numpy as np

from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

# A function that describe how pytorch to generate a representation of batch.
# This function if needed if the dataset return a data structure that the pytorch cannot recognize, e.g., the TreeNode above.
def collate_fn(batch):
    # Extract trees and labels from the batch
    trees = [item["tree"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])

    # Return the batch as a dictionary
    return {"tree": trees, "labels": labels}

def forwarding(batch):
    trees = batch["tree"]
    labels = batch["labels"]
    logits, nodes, edges = model(trees)
    # node : [n_batches, nodes (variable), 1, 417) 
    nodes = [torch.concat(node) for node in nodes]
    outputs = []
    for node_features, edge_index in zip(nodes, edges):
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        # Forward pass for a single graph
        output = gcn_model(node_features, edge_index)  # Shape: [num_nodes, output_dim]
        
        outputs.append(output)

    # Stack the outputs for all graphs
    outputs = torch.stack(outputs).squeeze(1)  # Shape: [batch_size, output_dim]
    loss = criterion(outputs, labels)
    return outputs, labels, loss

class TreeGNN(torch.nn.Module):
    def __init__(self, hidden_dim=128, num_classes = 3):
        super(TreeGNN, self).__init__()
        
        # Define GNN layers
        self.conv1 = GCNConv(417, hidden_dim)  # Input size: 1 + 32*4 = 129
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        
        x = F.relu(self.bn1(self.conv1(x, edge_index)))  # Shape: [num_nodes, hidden_dim]
        x = F.relu(self.bn2(self.conv2(x, edge_index)))          # Shape: [num_nodes, output_dim]
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = x.mean(dim=0, keepdim=True)  # Shape: [1, hidden_dim]

        x = self.fc(x)
        return x


# model = TreeGNN(hidden_dim=256,  num_classes = 3)


n_classes = 3
tree_records_base_path = "../data/serialized_tree"
dataset_types = ["normal-retained", "seizure", "pre-epileptic"]
dataset = TreeDataset(dataset_types, tree_records_base_path)

dataset_splitter = RatioBasedDataSetSplitter(dataset)
splitted_dataset = dataset_splitter.split_dataset()

train_subset = splitted_dataset["train_set"]
val_subset = splitted_dataset["val_set"]
test_subset = splitted_dataset["test_set"]

train_labels = [dataset.labels[i] for i in splitted_dataset["train_set_indexes"]]

sampler_builder = ProportionalWeightedRandomSamplerBuilder()
sampler, weights = sampler_builder.build(n_classes = 3, labels = train_labels, return_weights = True)

data_loader_builder = DataLoaderBuilder()
data_loaders = data_loader_builder.build(train_subset, val_subset, test_subset, train_sampler = sampler, batch_size = 32,
                         collate_fn_train = collate_fn, collate_fn_val = collate_fn, collate_fn_test = collate_fn)

train_loader = data_loaders["train_loader"]
val_loader = data_loaders["val_loader"]
test_loader = data_loaders["test_loader"]



# Hyperparameters
input_size = 32 * 3 + 64 + 1  # Size of the node value tuple
hidden_size = 128
num_classes = 3  # Normal, seizure, pre-epileptic
learning_rate = 0.0005
num_epochs = 10

# Initialize model, loss function, and optimizer
embedding_model = SeizurePredictionInputEmbeddingPreprocessor(unique_symbols = 96, \
                                                              symbol_embedding_size = 32, \
                                                              unique_grammar = 182, \
                                                              grammar_embedding_size = 64)

model = BinaryTreeLSTM(input_size, hidden_size, num_classes, input_embedding_model = embedding_model)
gcn_model = TreeGNN(hidden_dim=128,  num_classes = 3)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

enable_summary_confusion_matrix = True

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    if enable_summary_confusion_matrix:
        all_preds = []
        all_labels = []
    
    for batch in tqdm(train_loader):
        # Forward pass
        logits, labels, loss = forwarding(batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if enable_summary_confusion_matrix:
            # Store predictions and labels for metrics
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if enable_summary_confusion_matrix:
        # Calculate training metrics
        train_conf_matrix = confusion_matrix(all_labels, all_preds)
        train_metrics = calculate_metrics(train_conf_matrix)
    
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
        print(f"Training Metrics: {train_metrics}")
    print(f"Training Loss: {total_loss:.4f}")
    
    if enable_summary_confusion_matrix:
        print(f"Training Metrics: {train_metrics}")
    
    # Validation
    model.eval()
    val_loss = 0
    if enable_summary_confusion_matrix:
        val_preds = []
        val_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader):
            trees = batch["tree"]
            labels = batch["labels"]
    
            # Forward pass
            logits, labels, loss = forwarding(batch)
            val_loss += loss.item()
            
            if enable_summary_confusion_matrix:
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
    
    if enable_summary_confusion_matrix:
        val_conf_matrix = confusion_matrix(val_labels, val_preds)
        val_metrics = calculate_metrics(val_conf_matrix)
        
    print(f"Validation Loss: {val_loss:.4f}")
    
    if enable_summary_confusion_matrix:
        print(f"Validation Metrics: {val_metrics}")
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")