class Node:
    def __init__(self, id):
        self.id = id
        self.left = None
        self.right = None
        self.dist = None  # distance to parent
        self.count = 1  # leaf count

    def to_string(self, indent=""):
        s = indent + repr(self) + "\n"
        if self.left:
            s += self.left.to_string(indent=indent + "    ")
        if self.right:
            s += self.right.to_string(indent=indent + "    ")
        return s

    def __str__(self):
        return self.to_string().strip()

    def __repr__(self):
        return f"Node(id={self.id}, dist={self.dist}, count={self.count})"

    @property
    def is_leaf(self):
        return self.count == 1

    @property
    def children(self):
        if not self.is_leaf:
            return (self.left, self.right)


def to_tree(Z, rd=False):
    """TODO"""
    n_internal = Z.shape[0]
    n_original = n_internal + 1
    n_nodes = n_original + n_internal
    nodes = [Node(i) for i in range(n_nodes)]

    for i in range(n_internal):
        # Access the current parent node.
        parent_id = n_original + i
        parent = nodes[parent_id]

        # Access data about the children.
        left_id, right_id, left_dist, right_dist, leaf_count = Z[i]
        left_id = int(left_id)
        right_id = int(right_id)
        left = nodes[left_id]
        right = nodes[right_id]

        # Assign the children and leaf count.
        parent.left = left
        parent.right = right
        parent.count = int(leaf_count)

        # Set distances for the children
        left.dist = float(left_dist)
        right.dist = float(right_dist)

    root = nodes[-1]

    if rd:
        return root, nodes
    else:
        return root