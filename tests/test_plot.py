import plotly.graph_objects as go
import anjl


def test_example_1():
    # This example comes from Amelia Harrison's blog.
    # https://www.tenderisthebyte.com/blog/2022/08/31/neighbor-joining-trees/
    D, _ = anjl.data.example_1()
    Z = anjl.canonical_nj(D)
    fig = anjl.plot_equal_angle(Z)
    assert isinstance(fig, go.Figure)


def test_wikipedia_example():
    # This example comes from the wikipedia page on neighbour-joining.
    # https://en.wikipedia.org/wiki/Neighbor_joining#Example
    D, _ = anjl.data.wikipedia_example()
    Z = anjl.canonical_nj(D)
    fig = anjl.plot_equal_angle(Z)
    assert isinstance(fig, go.Figure)
