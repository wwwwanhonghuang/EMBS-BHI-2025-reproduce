
from data_structures.tree import SyntaxTreeNode
import torch
from torch_geometric.data import Data


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
        # value = (
        #     int(tokens[index]) & 0x0FFF,     # std::get<0>
        #     int(tokens[index + 1]) & 0x0FFF, # std::get<1>
        #     int(tokens[index + 2]) & 0x0FFF, # std::get<2>
        #     int(tokens[index + 3]), # std::get<3>
        #     float(tokens[index + 4]), # std::get<4>
        #     int(tokens[index + 5]) & 0x0FFF # std::get<5>
        # )
        def normalize_symbol(sym):
            return sym & 0x0FFF if sym != 0xFFFF else 0
        
        value = (
            normalize_symbol(int(tokens[index])),     # std::get<0>
            normalize_symbol(int(tokens[index + 1])), # std::get<1>
            normalize_symbol(int(tokens[index + 2])), # std::get<2>
            int(tokens[index + 3]), # std::get<3>
            float(tokens[index + 4]), # std::get<4>
            normalize_symbol(int(tokens[index + 5])) # std::get<5>
        )
        index += 6

        # Recursively deserialize left and right children
        left = _deserialize_helper()
        right = _deserialize_helper()

        return SyntaxTreeNode(value, left, right)

    return _deserialize_helper()




def tree_to_graph(root, device = 'cpu'):
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
        x.append(node_features.to(device))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)  # Edge indices

    return Data(
        x=torch.stack(x),  # Node feature matrix
        edge_index=edge_index # Edge index
    )

