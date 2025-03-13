class SyntaxTreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
    @property
    def children(self):
        return [self.left, self.right]

    def to_dict(self):
        left = {} if self.left is None else self.left.to_dict()
        right = {} if self.right is None else self.right.to_dict()
        return {"v": self.value, "l": left, "r": right}

        
    def __repr__(self):
        return f"SyntaxTreeNode(value={self.value}, left={self.left}, right={self.right})"
