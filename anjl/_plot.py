import math
from typing import Literal, Any
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from ._tree import Node
from ._layout import layout_equal_angle


def plot_equal_angle(
    tree: Node,
    leaf_data: pd.DataFrame | None = None,
    color: Any = None,
    symbol: Any = None,
    center_x: int | float = 0,
    center_y: int | float = 0,
    arc_start: int | float = 0,
    arc_stop: int | float = 2 * math.pi,
    distance_sort: bool = False,
    count_sort: bool = False,
    line_width: int | float = 1,
    marker_size: int | float = 5,
    width: int | float = 700,
    height: int | float = 600,
    render_mode: Literal["auto", "svg", "webgl"] = "auto",
    legend_sizing: Literal["constant", "trace"] = "constant",
) -> go.Figure:
    _, df_leaf_nodes, df_edges = layout_equal_angle(
        tree=tree,
        center_x=center_x,
        center_y=center_y,
        arc_start=arc_start,
        arc_stop=arc_stop,
        distance_sort=distance_sort,
        count_sort=count_sort,
    )

    # TODO Color the edges.
    # TODO Support hover name.
    # TODO Support hover data.

    # Decorate the leaf nodes.
    if leaf_data is not None:
        df_leaf_nodes = (
            df_leaf_nodes.set_index("id").join(leaf_data, how="left").reset_index()
        )

    # Draw the edges.
    fig1 = px.line(
        data_frame=df_edges,
        x="x",
        y="y",
        hover_name=None,
        hover_data=None,
        render_mode=render_mode,
    )

    # Draw the leaves.
    fig2 = px.scatter(
        data_frame=df_leaf_nodes,
        x="x",
        y="y",
        hover_name="id",
        hover_data=None,
        color=color,
        symbol=symbol,
        render_mode=render_mode,
    )

    # Combine traces into a single figure.
    fig = go.Figure()
    fig.add_traces(list(fig1.select_traces()))
    fig.add_traces(list(fig2.select_traces()))

    # Style lines and markers.
    line_props = dict(width=line_width)
    marker_props = dict(size=marker_size)
    fig.update_traces(line=line_props, marker=marker_props)

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
