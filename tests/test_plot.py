import plotly.graph_objects as go
import numpy as np
import anjl


def test_example_1():
    # This example comes from Amelia Harrison's blog.
    # https://www.tenderisthebyte.com/blog/2022/08/31/neighbor-joining-trees/
    D, _ = anjl.data.example_1()
    Z = anjl.canonical_nj(D)
    fig = anjl.plot(Z)
    assert isinstance(fig, go.Figure)


def test_wikipedia_example():
    # This example comes from the wikipedia page on neighbour-joining.
    # https://en.wikipedia.org/wiki/Neighbor_joining#Example
    D, _ = anjl.data.wikipedia_example()
    Z = anjl.canonical_nj(D)
    fig = anjl.plot(Z)
    assert isinstance(fig, go.Figure)


def test_mosquitoes():
    D, leaf_data = anjl.data.mosquitoes()
    Z = anjl.rapid_nj(D)
    fig = anjl.plot(Z, leaf_data=leaf_data, color="location", symbol="taxon")
    assert isinstance(fig, go.Figure)


def test_gh38():
    # https://github.com/alimanfoo/anjl/issues/38
    D, leaf_data = anjl.data.mosquitoes()
    leaf_data.loc[0, "taxon"] = np.nan
    leaf_data.loc[1, "location"] = np.nan
    Z = anjl.rapid_nj(D)
    fig = anjl.plot(Z, leaf_data=leaf_data, color="location", symbol="taxon")
    assert isinstance(fig, go.Figure)
