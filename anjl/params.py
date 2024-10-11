from typing import TypeAlias, Annotated, Callable, Literal
from collections.abc import Mapping
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import plotly.graph_objects as go


D: TypeAlias = Annotated[
    NDArray,
    "A distance matrix in square form.",
]

Z: TypeAlias = Annotated[
    NDArray[np.float32],
    """
    A neighbour-joining tree encoded as a numpy array. Each row in the array contains
    data for one internal node in the tree, in the order in which they were created by
    the neighbour-joining algorithm. Within each row there are five values: left child
    node identifier, right child node identifier, distance to left child, distance to
    right child, total number of leaves. This data structure is similar to that returned
    by scipy's hierarchical clustering functions, except that here we have two distance
    values for each internal node rather than one because distances to the children may
    be different.
    """,
]

disallow_negative_distances: TypeAlias = Annotated[
    bool, "If True, set any negative distances to zero."
]

progress: TypeAlias = Annotated[
    Callable | None,
    """
    A function which will be used to wrap the main loop iterator to provide information
    on progress. E.g., could be tqdm.
    """,
]

progress_options: TypeAlias = Annotated[
    Mapping,
    """Any options to be passed into the progress function.""",
]

gc: TypeAlias = Annotated[
    int | None,
    """
    Number of iterations to perform between compacting data structures to remove any
    data corresponding to nodes that have been clustered.
    """,
]

leaf_data: TypeAlias = Annotated[
    pd.DataFrame | None,
    """
    A pandas DataFrame containing additional data about the original observations.
    Length of the dataframe should be the same as the size of each dimension of the
    distance matrix.
    """,
]

color: TypeAlias = Annotated[
    str | None,
    "Name of variable to use to color the markers.",
]

symbol: TypeAlias = Annotated[
    str | None,
    "Name of the variable to use to choose marker symbols.",
]

marker_size: TypeAlias = Annotated[int | float, "Leaf node marker size."]

internal_marker_size: TypeAlias = Annotated[int | float, "Internal node marker size."]

line_width: TypeAlias = Annotated[int | float, "Edge line width."]

default_line_color: TypeAlias = Annotated[
    str, "Line color to use for edges where descendants are different colors."
]

fig_width: TypeAlias = Annotated[
    int | float | None,
    "Figure width in pixels (px).",
]

fig_height: TypeAlias = Annotated[
    int | float | None,
    "Figure height in pixels (px).",
]

color_discrete_sequence: TypeAlias = Annotated[list | None, "A list of colours to use."]

color_discrete_map: TypeAlias = Annotated[
    Mapping | None, "An explicit mapping from values to colours."
]

category_order: TypeAlias = Annotated[
    list | Mapping | None,
    "Control the order in which values appear in the legend.",
]

count_sort: TypeAlias = Annotated[
    bool,
    """
    If True, for each internal node, the child with the minimum number of descendants is
    plotted first. Note distance_sort and count_sort cannot both be True.
    """,
]

distance_sort: TypeAlias = Annotated[
    bool,
    """
    If True, for each internal node, the child with the minimum distance is plotted
    first. Note distance_sort and count_sort cannot both be True.
    """,
]

plotly_figure: TypeAlias = Annotated[go.Figure, "A plotly figure."]

render_mode: TypeAlias = Annotated[
    Literal["auto", "svg", "webgl"],
    "The type of rendering backend to use. See also https://plotly.com/python/webgl-vs-svg/",
]

legend_sizing: TypeAlias = Annotated[
    Literal["constant", "trace"],
    """
    Determines if the legend items symbols scale with their corresponding
    "trace" attributes or remain "constant" independent of the symbol size
    on the graph.
    """,
]

center_x: TypeAlias = Annotated[int | float, "X coordinate where plotting is centered."]

center_y: TypeAlias = Annotated[int | float, "Y coordinate where plotting is centered."]

arc_start: TypeAlias = Annotated[int | float, "Angle where tree layout begins."]

arc_stop: TypeAlias = Annotated[int | float, "Angle where tree layout ends."]

edge_legend: TypeAlias = Annotated[
    bool, "Show legend entries for the different edge (line) colors."
]

leaf_legend: TypeAlias = Annotated[
    bool,
    "Show legend entries for the different leaf node (scatter) colors and symbols.",
]

hover_name: TypeAlias = Annotated[
    str | None, "Name of variable to use as main label in hover tooltips."
]

hover_data: TypeAlias = Annotated[
    list[str] | None, "Names of addition variables to show in hover tooltips."
]
