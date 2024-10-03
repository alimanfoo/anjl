import math
import pandas as pd


def layout_equal_angle(
    tree,
    center_x=0,
    center_y=0,
    arc_start=0,
    arc_stop=2 * math.pi,
    distance_sort=False,
    count_sort=False,
):
    # Set up outputs.
    internal_nodes = []
    leaf_nodes = []
    edges = []

    # Begin recursion.
    _layout_equal_angle(
        node=tree,
        x=center_x,
        y=center_y,
        arc_start=arc_start,
        arc_stop=arc_stop,
        distance_sort=distance_sort,
        count_sort=count_sort,
        internal_nodes=internal_nodes,
        leaf_nodes=leaf_nodes,
        edges=edges,
    )

    # Load results into dataframes.
    df_internal_nodes = pd.DataFrame.from_records(
        internal_nodes, columns=["x", "y", "id"]
    )
    df_leaf_nodes = pd.DataFrame.from_records(leaf_nodes, columns=["x", "y", "id"])
    df_edges = pd.DataFrame.from_records(edges, columns=["x", "y"])

    return df_internal_nodes, df_leaf_nodes, df_edges


def _layout_equal_angle(
    *,
    node,
    x,
    y,
    arc_start,
    arc_stop,
    distance_sort,
    count_sort,
    internal_nodes,
    leaf_nodes,
    edges,
):
    if node.children:
        # Store internal node coordinates.
        internal_nodes.append([x, y, node.id])

        # Count leaves (descendants).
        leaf_count = node.count

        # Sort the subtrees.
        if distance_sort:
            children = sorted(node.children, key=lambda c: c.dist)
        elif count_sort:
            children = sorted(node.children, key=lambda c: c.count)
        else:
            children = node.children

        # Iterate over children, dividing up the current arc into
        # segments of size proportional to the number of leaves in
        # the subtree.
        arc_size = arc_stop - arc_start
        child_arc_start = arc_start
        for child in children:
            # Define a segment of the arc for this child.
            child_arc_size = arc_size * child.count / leaf_count
            child_arc_stop = child_arc_start + child_arc_size

            # Define the angle at which this child will be drawn.
            child_angle = child_arc_start + child_arc_size / 2

            # Access the distance at which to draw this child.
            distance = child.dist

            # Now use trigonometry to calculate coordinates for this child.
            child_x = x + distance * math.sin(child_angle)
            child_y = y + distance * math.cos(child_angle)

            # Add edge.
            edges.append([x, y])
            edges.append([child_x, child_y])
            edges.append([None, None])

            # Recurse to layout the child.
            _layout_equal_angle(
                node=child,
                x=child_x,
                y=child_y,
                internal_nodes=internal_nodes,
                leaf_nodes=leaf_nodes,
                edges=edges,
                arc_start=child_arc_start,
                arc_stop=child_arc_stop,
                count_sort=count_sort,
                distance_sort=distance_sort,
            )

            # Update loop variables ready for the next iteration.
            child_arc_start = child_arc_stop

    else:
        leaf_nodes.append([x, y, node.id])
