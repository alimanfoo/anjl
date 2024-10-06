import math
from itertools import cycle
from typing import Literal, Any
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from ._layout import layout_equal_angle
from ._util import decorate_internal_nodes


def plot_equal_angle(
    Z: np.ndarray,
    leaf_data: pd.DataFrame | None = None,
    color: Any = None,
    symbol: Any = None,
    hover_name: str | None = None,
    hover_data: list | None = None,
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
    internal_legend: bool = False,
    default_edge_color="black",
    width: int | float = 700,
    height: int | float = 600,
    render_mode: Literal["auto", "svg", "webgl"] = "auto",
    legend_sizing: Literal["constant", "trace"] = "constant",
) -> go.Figure:
    df_internal_nodes, df_leaf_nodes, df_edges = layout_equal_angle(
        Z=Z,
        center_x=center_x,
        center_y=center_y,
        arc_start=arc_start,
        arc_stop=arc_stop,
        distance_sort=distance_sort,
        count_sort=count_sort,
    )

    # TODO Color the edges.
    # TODO Support hover_name.
    # TODO Support hover_data.
    # TODO Support color_discrete_map.
    # TODO Support category_orders (ordering the legend).
    # TODO Support edge_legend.
    # TODO Support leaf_legend.

    # Decorate the plot.
    if leaf_data is not None:
        df_leaf_nodes = (
            df_leaf_nodes.set_index("id").join(leaf_data, how="left").reset_index()
        )
        if color is not None:
            leaf_color_values = leaf_data[color].values
            unique_color_values = np.unique(leaf_color_values)
            if category_orders is None:
                category_orders = {color: unique_color_values}
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
            color_discrete_map[""] = default_edge_color
            internal_color_values = decorate_internal_nodes(Z, leaf_color_values)
            color_values = np.concatenate([leaf_color_values, internal_color_values])
            color_data = pd.DataFrame({color: color_values})
            df_edges = df_edges.join(color_data, on="id", how="left")
            df_internal_nodes = df_internal_nodes.join(color_data, on="id", how="left")

    # Combine traces into a single figure.
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

    # Draw the leaves.
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
        fig3.update_traces(marker=internal_marker_props, showlegend=internal_legend)
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
