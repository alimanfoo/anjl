import numpy as np
import plotly.graph_objects as go
import anjl


def test_amelia_harrison_example():
    # This example comes from Amelia Harrison's blog.
    # https://www.tenderisthebyte.com/blog/2022/08/31/neighbor-joining-trees/

    D = np.array(
        [  # A B C D
            [0, 4, 5, 10],
            [4, 0, 7, 12],
            [5, 7, 0, 9],
            [10, 12, 9, 0],
        ],
        dtype=np.float32,
    )
    Z = anjl.canonical_nj(D)
    fig = anjl.plot_equal_angle(Z)
    assert isinstance(fig, go.Figure)


def test_wikipedia_example():
    # This example comes from the wikipedia page on neighbour-joining.
    # https://en.wikipedia.org/wiki/Neighbor_joining#Example

    D = np.array(
        [  # a b c d e
            [0, 5, 9, 9, 8],
            [5, 0, 10, 10, 9],
            [9, 10, 0, 8, 7],
            [9, 10, 8, 0, 3],
            [8, 9, 7, 3, 0],
        ],
        dtype=np.float32,
    )
    Z = anjl.canonical_nj(D)
    fig = anjl.plot_equal_angle(Z)
    assert isinstance(fig, go.Figure)
