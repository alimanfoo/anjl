from typing import TypeAlias, Annotated, Callable
from collections.abc import Mapping
import numpy as np
from numpy.typing import NDArray


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
    right child, total number of leaves.

    This data structure is similar to that returned by scipy's hierarchical clustering
    functions, except that here we have two distance values for each internal node
    rather than one because distances to the children may be different.

    Please note that the ordering of the internal nodes may be different between the
    canonical and the rapid algorithms, because these algorithms search the distance
    matrix in a different order. However, the resulting trees will be topologically
    equivalent.
    """,
]


disallow_negative_distances: TypeAlias = Annotated[
    bool, "If True, set any negative distances to zero."
]


progress: TypeAlias = Annotated[
    Callable | None,
    """
    A function which will be used to wrap the main loop iterator. E.g., could be tqdm.
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
