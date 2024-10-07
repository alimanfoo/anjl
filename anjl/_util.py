import numpy as np


def to_string(Z: np.ndarray) -> str:
    """TODO"""
    # Total number of internal nodes.
    n_internal = Z.shape[0]

    # Total number of leaf nodes.
    n_original = n_internal + 1

    # Set up the first node to visit, which will be the root node.
    root = n_original + n_internal - 1

    # Initialise working variables.
    text = ""
    stack = [(root, 0, "")]

    # Start processing nodes.
    while stack:
        # Access the next node to process.
        node, dist, indent = stack.pop()
        if node < n_original:
            # Leaf node.
            text += f"{indent}Leaf(id={node}, dist={dist})\n"
        else:
            # Internal node.
            z = node - n_original
            left = int(Z[z, 0])
            right = int(Z[z, 1])
            ldist = Z[z, 2]
            rdist = Z[z, 3]
            count = int(Z[z, 4])
            text += f"{indent}Node(id={node}, dist={dist}, count={count})\n"

            # Put them on the stack in this order so the left node comes out first.
            stack.append((right, rdist, indent + "    "))
            stack.append((left, ldist, indent + "    "))

    return text.strip()


def map_internal_to_leaves(Z: np.ndarray) -> list[list[int]]:
    """TODO"""

    # For each internal node, build a list of all the descendant leaf ids.
    index: list[list[int]] = []

    # Total number of internal nodes.
    n_internal = Z.shape[0]

    # Total number of leaf nodes.
    n_original = n_internal + 1

    # Iterate over internal nodes.
    for z in range(n_internal):
        # Create a list to store the leaves for this node.
        leaves = []

        # Access the direct children.
        left = int(Z[z, 0])
        right = int(Z[z, 1])

        # Add to the leaves.
        if left < n_original:
            leaves.append(left)
        else:
            leaves.extend(index[left - n_original])
        if right < n_original:
            leaves.append(right)
        else:
            leaves.extend(index[right - n_original])

        # Store the leaves in the index.
        index.append(leaves)

    return index
