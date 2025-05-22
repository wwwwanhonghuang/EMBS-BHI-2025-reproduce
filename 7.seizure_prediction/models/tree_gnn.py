
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

class TreeGNN(torch.nn.Module):
    def __init__(self, hidden_dim=128, num_classes = 3, use_classification_layer = True):
        super(TreeGNN, self).__init__()
        # Define embedding layers
        self.dim0_embedding = nn.Embedding(num_embeddings=96, embedding_dim=32)
        self.dim1_embedding = nn.Embedding(num_embeddings=96, embedding_dim=32)
        self.dim2_embedding = nn.Embedding(num_embeddings=96, embedding_dim=32)
        self.dim5_embedding = nn.Embedding(num_embeddings=256, embedding_dim=32)

        # Define GNN layers
        self.conv1 = GCNConv(129, hidden_dim)  # Input size: 1 + 32*4 = 129
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        if use_classification_layer:
            self.fc = nn.Linear(hidden_dim, num_classes)
        else:
            self.fc = None
        self.use_classification_layer = use_classification_layer


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

        if self.use_classification_layer:
            x = self.fc(x)
        return x
    
    
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class HighOrderTreeSequentialGCNModel(nn.Module):
    def __init__(self, 
                 gcn_hidden_dim=256,
                 lstm_hidden_dim=128,
                 num_classes=3,
                 num_lstm_layers=2,
                 dropout=0.2):
        super(HighOrderTreeSequentialGCNModel, self).__init__()
        
        # TreeGNN module (without final classification layer)
        self.tree_gnn = TreeGNN(
            hidden_dim=gcn_hidden_dim,
            num_classes=num_classes,
            use_classification_layer=False  # We'll do classification after LSTM
        )
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=gcn_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim//2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, graphs):
        """
        Args:
            batch: Dictionary containing:
                - graphs: List of PyG Data objects (sequence of trees)
                - label: Ground truth label
        Returns:
            logits: Classification logits
        """
        # Process each tree in the sequence with GNN
        tree_features = []
        for graph in graphs:
            # Extract features from individual tree
            x = self.tree_gnn(
                x=graph.x,
                edge_index=graph.edge_index,
                batch=graph.batch
            )
            tree_features.append(x)
        
        # Stack features along sequence dimension [batch, seq_len, features]
        sequence = torch.stack(tree_features, dim=1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(sequence)
        
        # Get final sequence output
        seq_features = lstm_out[:, -1, :]  # Take last timestep output
        
        # Classification
        logits = self.classifier(self.dropout(seq_features))
        
        return logits