import math
from itertools import cycle
from typing import Literal
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from ._layout import layout_equal_angle


def plot_equal_angle(
    Z: np.ndarray,
    leaf_data: pd.DataFrame | None = None,
    color: str | None = None,
    symbol: str | None = None,
    hover_name: str | None = None,
    hover_data: list[str] | None = None,
    center_x: int | float = 0,
    center_y: int | float = 0,
    arc_start: int | float = 0,
    arc_stop: int | float = 2 * math.pi,
    count_sort: bool = True,
    distance_sort: bool = False,
    line_width: int | float = 1,
    marker_size: int | float = 5,
    internal_marker_size: int | float = 0,
    color_discrete_sequence=None,
    color_discrete_map=None,
    category_orders=None,
    leaf_legend: bool = True,
    edge_legend: bool = False,
    default_edge_color="black",
    width: int | float = 700,
    height: int | float = 600,
    render_mode: Literal["auto", "svg", "webgl"] = "auto",
    legend_sizing: Literal["constant", "trace"] = "constant",
) -> go.Figure:
    """TODO"""
    # Layout the nodes in the tree according to the equal angles algorithm.
    df_internal_nodes, df_leaf_nodes, df_edges = layout_equal_angle(
        Z=Z,
        center_x=center_x,
        center_y=center_y,
        arc_start=arc_start,
        arc_stop=arc_stop,
        distance_sort=distance_sort,
        count_sort=count_sort,
    )

    # Don't plot a legend unless there are color and/or symbol options.
    if color is None and symbol is None:
        leaf_legend = edge_legend = False

    # Decorate the tree.
    df_internal_nodes, df_leaf_nodes, df_edges = decorate_tree(
        Z=Z,
        df_internal_nodes=df_internal_nodes,
        df_leaf_nodes=df_leaf_nodes,
        df_edges=df_edges,
        leaf_data=leaf_data,
        color=color,
    )

    # Populate and normalise color parameters.
    category_orders, color_discrete_sequence, color_discrete_map = (
        normalise_color_params(
            leaf_data=leaf_data,
            color=color,
            category_orders=category_orders,
            color_discrete_sequence=color_discrete_sequence,
            color_discrete_map=color_discrete_map,
            default_edge_color=default_edge_color,
        )
    )

    # Create a single figure that will be used for all traces.
    fig = go.Figure()

    # Draw the edges.
    fig1 = px.line(
        data_frame=df_edges,
        x="x",
        y="y",
        hover_name=None,
        hover_data=None,
        color=color,
        category_orders=category_orders,
        color_discrete_map=color_discrete_map,
        color_discrete_sequence=color_discrete_sequence,
        render_mode=render_mode,
    )
    line_props = dict(width=line_width)
    fig1.update_traces(line=line_props, showlegend=edge_legend)
    fig.add_traces(list(fig1.select_traces()))

    # Draw the leaf nodes.
    if hover_name is None:
        hover_name = "id"
    fig2 = px.scatter(
        data_frame=df_leaf_nodes,
        x="x",
        y="y",
        hover_name=hover_name,
        hover_data=hover_data,
        color=color,
        category_orders=category_orders,
        color_discrete_map=color_discrete_map,
        color_discrete_sequence=color_discrete_sequence,
        symbol=symbol,
        render_mode=render_mode,
    )
    marker_props = dict(size=marker_size)
    fig2.update_traces(marker=marker_props, showlegend=leaf_legend)
    fig.add_traces(list(fig2.select_traces()))

    if internal_marker_size > 0:
        # Draw the internal nodes.
        fig3 = px.scatter(
            data_frame=df_internal_nodes,
            x="x",
            y="y",
            hover_name="id",
            color=color,
            category_orders=category_orders,
            color_discrete_map=color_discrete_map,
            color_discrete_sequence=color_discrete_sequence,
            render_mode=render_mode,
        )
        internal_marker_props = dict(size=internal_marker_size)
        fig3.update_traces(marker=internal_marker_props, showlegend=False)
        fig.add_traces(list(fig3.select_traces()))

    # Style the figure.
    fig.update_layout(
        width=width,
        height=height,
        template="simple_white",
        legend=dict(itemsizing=legend_sizing, tracegroupgap=0),
    )

    # Style the axes.
    fig.update_xaxes(
        title=None,
        mirror=False,
        showgrid=False,
        showline=False,
        showticklabels=False,
        ticks="",
    )
    fig.update_yaxes(
        title=None,
        mirror=False,
        showgrid=False,
        showline=False,
        showticklabels=False,
        ticks="",
        # N.B., this is important, as it prevents distortion of the tree.
        # See also https://plotly.com/python/axes/#fixed-ratio-axes
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


def normalise_color_params(
    leaf_data: pd.DataFrame | None,
    color: str | None,
    category_orders,
    color_discrete_sequence,
    color_discrete_map,
    default_edge_color,
):
    """TODO"""
    if leaf_data is not None and color is not None:
        # We need all traces to use the same color configuration, and so we population
        # and normalise the category_orders, color_discrete_sequence and
        # color_discrete_map parameters.

        # Access the leaf colors.
        leaf_color_values = np.asarray(leaf_data[color].values)

        # Find all unique color values.
        unique_color_values = np.unique(leaf_color_values)

        # Normalise category orders.
        if category_orders is None:
            category_orders = {color: unique_color_values}

        # Normalise the color mapping.
        if color_discrete_map is None:
            if color_discrete_sequence is None:
                if len(unique_color_values) <= 10:
                    color_discrete_sequence = px.colors.qualitative.Plotly
                else:
                    color_discrete_sequence = px.colors.qualitative.Alphabet
            # Map values to colors.
            color_discrete_map = {
                v: c
                for v, c in zip(unique_color_values, cycle(color_discrete_sequence))
            }

        # Set a default color for edges where descendant leaves have
        # different colors.
        color_discrete_map[""] = default_edge_color

    return category_orders, color_discrete_sequence, color_discrete_map


def decorate_tree(
    Z: np.ndarray,
    df_internal_nodes: pd.DataFrame,
    df_leaf_nodes: pd.DataFrame,
    df_edges: pd.DataFrame,
    leaf_data: pd.DataFrame | None,
    color: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """TODO"""

    if leaf_data is not None:
        # Join the leaf data with the leaf nodes, so leaf data is
        # available for color, symbol, hover_name and hover_data
        # options.
        df_leaf_nodes = (
            df_leaf_nodes.set_index("id").join(leaf_data, how="left").reset_index()
        )

        # Further handling of color parameter if provided.
        if color is not None:
            # Access the leaf colors.
            leaf_color_values = np.asarray(leaf_data[color].values)

            # Associate colors with internal nodes and edges.
            internal_color_values = paint_internal(Z, leaf_color_values)
            color_values = np.concatenate([leaf_color_values, internal_color_values])
            color_data = pd.DataFrame({color: color_values})
            df_edges = df_edges.join(color_data, on="id", how="left")
            df_internal_nodes = df_internal_nodes.join(color_data, on="id", how="left")

    return df_internal_nodes, df_leaf_nodes, df_edges


def paint_internal(Z: np.ndarray, leaf_color_values: np.ndarray) -> np.ndarray:
    """TODO"""

    # For each internal node, create a set to store the colors
    # associated with all leaf nodes.
    internal_color_sets: list[set] = []

    # For each internal node, assign either a single color if
    # all leaf nodes have the same color, or a missing value.
    internal_color_values = np.empty(shape=Z.shape[0], dtype=leaf_color_values.dtype)

    # Total number of internal nodes.
    n_internal = Z.shape[0]

    # Total number of leaf nodes.
    n_original = n_internal + 1

    # Iterate over internal nodes.
    for z in range(n_internal):
        # Create a set to store the color values for this node.
        colors = set()

        # Access the direct children of this node.
        left = int(Z[z, 0])
        right = int(Z[z, 1])

        # Handle the left child.
        if left < n_original:
            # Child is a leaf node.
            colors.add(leaf_color_values[left])
        else:
            # Child is an internal node.
            colors.update(internal_color_sets[left - n_original])

        # Handle the right child.
        if right < n_original:
            # Child is a leaf node.
            colors.add(leaf_color_values[right])
        else:
            # Child is an internal node.
            colors.update(internal_color_sets[right - n_original])

        # Store a singleton value if present.
        if len(colors) == 1:
            # All leaves have the same color, paint this internal node.
            internal_color_values[z] = list(colors)[0]
        else:
            # Leaves have different colors, do not paint this internal node.
            internal_color_values[z] = ""

        # Store all the values for use in subsequent interations.
        internal_color_sets.append(colors)

    return internal_color_values
