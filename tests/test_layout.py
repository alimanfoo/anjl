import pandas as pd
import anjl


def validate_layout_result(D, Z, df_internal_nodes, df_leaf_nodes, df_edges):
    n_original = D.shape[0]
    n_internal = Z.shape[0]

    # Check the internal nodes.
    assert isinstance(df_internal_nodes, pd.DataFrame)
    assert len(df_internal_nodes) == n_internal
    assert df_internal_nodes.columns.to_list() == ["x", "y", "id"]

    # Check the leaf nodes.
    assert isinstance(df_leaf_nodes, pd.DataFrame)
    assert len(df_leaf_nodes) == n_original
    assert df_leaf_nodes.columns.to_list() == ["x", "y", "id"]

    # Check the edges.
    assert isinstance(df_edges, pd.DataFrame)
    # 2 edges per internal node, three rows per edge to get each
    # endpoint plus null row to get plotly to break between edges.
    assert len(df_edges) == 6 * n_internal
    assert df_edges.columns.to_list() == ["x", "y", "id"]


def test_example_1():
    # This example comes from Amelia Harrison's blog.
    # https://www.tenderisthebyte.com/blog/2022/08/31/neighbor-joining-trees/
    D, _ = anjl.data.example_1()
    Z = anjl.canonical_nj(D)
    df_internal_nodes, df_leaf_nodes, df_edges = anjl.layout_equal_angle(Z)
    validate_layout_result(
        D=D,
        Z=Z,
        df_internal_nodes=df_internal_nodes,
        df_leaf_nodes=df_leaf_nodes,
        df_edges=df_edges,
    )


def test_wikipedia_example():
    # This example comes from the wikipedia page on neighbour-joining.
    # https://en.wikipedia.org/wiki/Neighbor_joining#Example
    D, _ = anjl.data.wikipedia_example()
    Z = anjl.canonical_nj(D)
    df_internal_nodes, df_leaf_nodes, df_edges = anjl.layout_equal_angle(Z)
    validate_layout_result(
        D=D,
        Z=Z,
        df_internal_nodes=df_internal_nodes,
        df_leaf_nodes=df_leaf_nodes,
        df_edges=df_edges,
    )
