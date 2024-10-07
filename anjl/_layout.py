import math
import numpy as np
import pandas as pd


def layout_equal_angle(
    Z: np.ndarray,
    center_x: int | float = 0,
    center_y: int | float = 0,
    arc_start: int | float = 0,
    arc_stop: int | float = 2 * math.pi,
    distance_sort: bool = False,
    count_sort: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """TODO"""

    # Set up outputs.
    internal_nodes: list[tuple] = []
    leaf_nodes: list[tuple] = []
    edges: list[tuple] = []

    # Total number of internal nodes.
    n_internal = Z.shape[0]

    # Total number of leaf nodes.
    n_original = n_internal + 1

    # Set up the first node to visit, which will be the
    # root node.
    root = n_original + n_internal - 1

    # Initialise the stack, first task is to position the root node.
    stack = [
        (
            root,
            float(center_x),
            float(center_y),
            float(arc_start),
            float(arc_stop),
        )
    ]

    # Start processing.
    while stack:
        # Access the next node to process.
        (node, node_x, node_y, node_arc_start, node_arc_stop) = stack.pop()

        # Process the node.
        _layout_equal_angle(
            Z=Z,
            leaf_nodes=leaf_nodes,
            internal_nodes=internal_nodes,
            edges=edges,
            distance_sort=distance_sort,
            count_sort=count_sort,
            stack=stack,
            node=node,
            x=node_x,
            y=node_y,
            arc_start=node_arc_start,
            arc_stop=node_arc_stop,
        )

    # Load results into dataframes.
    df_internal_nodes = pd.DataFrame.from_records(
        internal_nodes, columns=["x", "y", "id"]
    )
    df_leaf_nodes = pd.DataFrame.from_records(leaf_nodes, columns=["x", "y", "id"])
    df_edges = pd.DataFrame.from_records(edges, columns=["x", "y", "id"])

    return df_internal_nodes, df_leaf_nodes, df_edges


def _layout_equal_angle(
    *,
    node: int,
    Z: np.ndarray,
    leaf_nodes: list[tuple],
    internal_nodes: list[tuple],
    edges: list[tuple],
    distance_sort: bool,
    count_sort: bool,
    stack: list[tuple],
    x: float,
    y: float,
    arc_start: float,
    arc_stop: float,
) -> None:
    # Total number of internal nodes.
    n_internal = Z.shape[0]

    # Total number of leaf nodes.
    n_original = n_internal + 1

    if node < n_original:
        # Leaf node.
        leaf_nodes.append((x, y, node))

    else:
        # Internal node.
        z = node - n_original

        # Access data for this node and its children.
        left = int(Z[z, 0])
        right = int(Z[z, 1])
        dist_l = Z[z, 2]
        dist_r = Z[z, 3]
        leaf_count = int(Z[z, 4])
        if left < n_original:
            count_l = 1
        else:
            count_l = int(Z[left - n_original, 4])
        if right < n_original:
            count_r = 1
        else:
            count_r = int(Z[right - n_original, 4])

        # Store internal node coordinates.
        internal_nodes.append((x, y, node))

        # Set up convenience variable.
        children = [(left, dist_l, count_l), (right, dist_r, count_r)]

        # Sort the subtrees.
        if distance_sort and dist_r < dist_l:
            children.reverse()
        elif count_sort and count_r < count_l:
            children.reverse()

        # Iterate over children, dividing up the current arc into
        # segments of size proportional to the number of leaves in
        # the subtree.
        arc_size = arc_stop - arc_start
        child_arc_start = arc_start
        for child, child_dist, child_count in children:
            # Define a segment of the arc for this child.
            child_arc_size = arc_size * child_count / leaf_count
            child_arc_stop = child_arc_start + child_arc_size

            # Define the angle at which this child will be drawn.
            child_angle = child_arc_start + child_arc_size / 2

            # Now use trigonometry to calculate coordinates for this child.
            child_x = x + child_dist * math.sin(child_angle)
            child_y = y + child_dist * math.cos(child_angle)

            # Add edge.
            edges.append((x, y, child))
            edges.append((child_x, child_y, child))
            edges.append((None, None, child))

            # Add a task to layout the child.
            stack.append(
                (
                    child,
                    child_x,
                    child_y,
                    child_arc_start,
                    child_arc_stop,
                )
            )

            # Update loop variables ready for the next child.
            child_arc_start = child_arc_stop
