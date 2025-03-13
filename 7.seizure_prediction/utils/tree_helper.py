
from data_structures.tree import SyntaxTreeNode

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
        value = [
            int(tokens[index]),     # std::get<0>
            int(tokens[index + 1]), # std::get<1>
            int(tokens[index + 2]), # std::get<2>
            int(tokens[index + 3]), # std::get<3>
            float(tokens[index + 4]), # std::get<4>
            int(tokens[index + 5])  # std::get<5>
        ]
        value[0] = 0 if value[0] == 65535 else value[0] + 1
        value[1] = 0 if value[1] == 65535 else value[1] + 1
        value[2] = 0 if value[2] == 65535 else value[2] + 1
        value[5] = 0 if value[5] == 65535 else value[5] + 1

        index += 6

        # Recursively deserialize left and right children
        left = _deserialize_helper()
        right = _deserialize_helper()

        return SyntaxTreeNode(value, left, right)

    return _deserialize_helper()
