import torch
import torch.nn as nn
from joblib import Parallel, delayed

class BinaryTreeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryTreeLSTMCell, self).__init__()
        self.hidden_size = hidden_size

        self.input_gate = nn.Linear(input_size + hidden_size * 2, hidden_size)
        self.forget_gate_left = nn.Linear(input_size + hidden_size * 2, hidden_size)
        self.forget_gate_right = nn.Linear(input_size + hidden_size * 2, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size * 2, hidden_size)
        self.cell_state = nn.Linear(input_size + hidden_size * 2, hidden_size)

    def forward(self, x, h_left, c_left, h_right, c_right):
        """
        Args:
            x: Input feature vector of the current node (batch_size, input_size)
            h_left: Hidden state of the left child (batch_size, hidden_size)
            c_left: Cell state of the left child (batch_size, hidden_size)
            h_right: Hidden state of the right child (batch_size, hidden_size)
            c_right: Cell state of the right child (batch_size, hidden_size)
        Returns:
            h: Hidden state of the current node (batch_size, hidden_size)
            c: Cell state of the current node (batch_size, hidden_size)
        """
        # Concatenate input with children's hidden states
        h_children = torch.cat([h_left, h_right], dim=1) # 2x64 = 128
        combined = torch.cat([x, h_children], dim=1) # 128 + 6 = 134

        # Compute gates
        i = torch.sigmoid(self.input_gate(combined))  # Input gate
        f_left = torch.sigmoid(self.forget_gate_left(combined))  # Forget gate (left child)
        f_right = torch.sigmoid(self.forget_gate_right(combined))  # Forget gate (right child)
        o = torch.sigmoid(self.output_gate(combined))  # Output gate
        u = torch.tanh(self.cell_state(combined))  # Candidate cell state

        # Update cell state
        c = i * u + f_left * c_left + f_right * c_right

        # Update hidden state
        h = o * torch.tanh(c)

        return h, c
    
class BinaryTreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BinaryTreeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.tree_lstm_cell = BinaryTreeLSTMCell(input_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.num_classes = num_classes

    # def forward(self, trees):
    #     """
    #     Args:
    #         tree: A dictionary representing the tree (output from collate_fn)
    #     Returns:
    #         logits: Classification logits (batch_size, num_classes)
    #     """

    #     batch_logits = []
    #     for tree in trees:
    #         h, c = self._recursive_forward(tree)
    #         logits = self.classifier(h).squeeze(0)
    #         batch_logits.append(logits)
    #     results = torch.stack(batch_logits)
    #     return results
    def forward(self, trees):
        """
        Args:
            trees: A list of dictionaries representing the trees (output from collate_fn)
        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        def process_tree(tree):
            h, c = self._recursive_forward(tree)
            return self.classifier(h).squeeze(0)

        batch_logits = Parallel(n_jobs=-1)(delayed(process_tree)(tree) for tree in trees)
        results = torch.stack(batch_logits)
        return results


    def _recursive_forward(self, node):
        """
        Recursively process a node and its children.
        Args:
            node: A dictionary representing the current node
        Returns:
            h: Hidden state of the current node
            c: Cell state of the current node
        """
        if not node:  # Base case: leaf node
            h = torch.zeros(1, self.hidden_size)
            c = torch.zeros(1, self.hidden_size)
            
            return h, c

        # Process left and right children
        h_left, c_left = self._recursive_forward(node.get("l", {}))
        h_right, c_right = self._recursive_forward(node.get("r", {}))
        
        # Convert node value to a tensor (assuming node["v"] is a tuple of 6 values)
        x = torch.tensor(node["v"], dtype=torch.float32).unsqueeze(0)  # (1, input_size)

        # Apply Tree-LSTM cell
        h, c = self.tree_lstm_cell(x, h_left, c_left, h_right, c_right)

        return h, c