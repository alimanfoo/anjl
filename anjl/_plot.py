import plotly.express as px
import plotly.graph_objects as go
import math
from ._layout import layout_equal_angle


def plot_equal_angle(
    tree,
    center_x=0,
    center_y=0,
    arc_start=0,
    arc_stop=2 * math.pi,
    distance_sort=False,
    count_sort=False,
    line_width=1,
    marker_size=5,
    width=700,
    height=600,
    render_mode="auto",
):
    _, df_leaf_nodes, df_edges = layout_equal_angle(
        tree=tree,
        center_x=center_x,
        center_y=center_y,
        arc_start=arc_start,
        arc_stop=arc_stop,
        distance_sort=distance_sort,
        count_sort=count_sort,
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
