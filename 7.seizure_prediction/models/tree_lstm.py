import torch
import torch.nn as nn
from joblib import Parallel, delayed

class SeizurePredictionInputEmbeddingPreprocessor(nn.Module):
    def __init__(self, unique_symbols, symbol_embedding_size, unique_grammar, grammar_embedding_size):
        super(SeizurePredictionInputEmbeddingPreprocessor, self).__init__()
        self.symbol_embedding = nn.Embedding(unique_symbols, symbol_embedding_size)
        self.grammar_embedding = nn.Embedding(unique_grammar, grammar_embedding_size)

    def forward(self, v):
        v4_tensor = torch.tensor(v[4], dtype=torch.float32).unsqueeze(0)

        return torch.cat([
            self.symbol_embedding(torch.tensor(v[0], dtype=torch.long)),
            self.symbol_embedding(torch.tensor(v[1], dtype=torch.long)),
            self.symbol_embedding(torch.tensor(v[2], dtype=torch.long)),
            v4_tensor,
            self.grammar_embedding(torch.tensor(v[5], dtype=torch.long))
        ])
    
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
    def __init__(self, input_size, hidden_size, num_classes, input_embedding_model = None):
        super(BinaryTreeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.tree_lstm_cell = BinaryTreeLSTMCell(input_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.num_classes = num_classes
        self.embedding_model = input_embedding_model
        self.input_size = input_size

    # def forward(self, trees):
    #     """
    #     Args:
    #         trees: A list of dictionaries representing the trees (output from collate_fn)
    #     Returns:
    #         logits: Classification logits (batch_size, num_classes)
    #     """
    #     def process_tree(tree):
    #         h, c = self._recursive_forward(tree)
    #         return self.classifier(h).squeeze(0)

    #     batch_logits = [process_tree(tree) for tree in trees]
    #     results = torch.stack(batch_logits)
    #     return results

    def forward(self, trees):
        """
        Args:
            trees: A list of dictionaries representing the trees (output from collate_fn)
        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        def process_tree(tree, index):
            h, c, nodes, edges = self._recursive_forward(tree, index, [], [], None)
            return self.classifier(h).squeeze(0), nodes, edges

        n = len(trees)

        results = Parallel(n_jobs=16)(delayed(process_tree)(tree, index) for index, tree in enumerate(trees))
        batch_logits, nodes, edges = zip(*results)
        batch_logits = torch.stack(batch_logits)
        return batch_logits, nodes, edges




    def _recursive_forward(self, node, index, nodes, edges, parent_id = None):            
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
            new_node_id = len(nodes)
            node_features = self.embedding_model([0, 0, 0, 0, 0, 0]).view(1, -1)
            nodes.append(torch.cat([h, c, node_features], dim = 1))
            if parent_id is not None:
                edges.append([parent_id, new_node_id])
            return h, c, nodes, edges
    
        node_id = len(nodes)
        nodes.append(None)
        

        # Process left and right children
        h_left, c_left, nodes, edges = self._recursive_forward(node.get("l", {}), index, nodes, edges, node_id)
        h_right, c_right, nodes, edges = self._recursive_forward(node.get("r", {}), index, nodes, edges, node_id)
    
        v = node["v"]
        if self.embedding_model:
            x = self.embedding_model(v).view(1, self.input_size)
        else:
            x = torch.tensor(v, dtype=torch.float32).unsqueeze(0)  # (1, input_size)

        # Apply Tree-LSTM cell
        h, c = self.tree_lstm_cell(x, h_left, c_left, h_right, c_right)
        
        nodes[node_id] = torch.cat([h, c, x], dim = 1)
        if parent_id is not None:
            edges.append([parent_id, node_id])
        return h, c, nodes, edges
