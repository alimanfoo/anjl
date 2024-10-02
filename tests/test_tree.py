from textwrap import dedent
import numpy as np
from numpy.testing import assert_allclose
import anjl


def test_amelia_harrison():
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
    n_original = D.shape[0]
    n_internal = n_original - 1
    n_nodes = n_original + n_internal

    # Default options.
    root = anjl.to_tree(Z)
    assert isinstance(root, anjl.Node)
    assert root.id == n_nodes - 1
    assert root.dist is None
    assert root.count == n_original
    assert not root.is_leaf

    # First level.
    left = root.left
    right = root.right
    assert left.id == 3
    assert_allclose(left.dist, 3.5)
    assert left.count == 1
    assert left.is_leaf
    assert left.left is None
    assert left.right is None
    assert right.id == 5
    assert_allclose(right.dist, 3.5)
    assert right.count == 3
    assert not right.is_leaf
    assert right.left is not None
    assert right.right is not None

    # Second level.
    parent = right
    left = parent.left
    right = parent.right
    assert left.id == 2
    assert_allclose(left.dist, 2.0)
    assert left.count == 1
    assert left.is_leaf
    assert left.left is None
    assert left.right is None
    assert right.id == 4
    assert_allclose(right.dist, 2.0)
    assert right.count == 2
    assert not right.is_leaf
    assert right.left is not None
    assert right.right is not None

    # Third level.
    parent = right
    left = parent.left
    right = parent.right
    assert left.id == 0
    assert_allclose(left.dist, 1.0)
    assert left.count == 1
    assert left.is_leaf
    assert left.left is None
    assert left.right is None
    assert right.id == 1
    assert_allclose(right.dist, 3.0)
    assert right.count == 1
    assert right.is_leaf
    assert right.left is None
    assert right.right is None

    # String representation.
    expected_repr = "Node(id=6, dist=None, count=4)"
    assert repr(root) == expected_repr

    # String value.
    expected_str = dedent("""
        Node(id=6, dist=None, count=4)
            Node(id=3, dist=3.5, count=1)
            Node(id=5, dist=3.5, count=3)
                Node(id=2, dist=2.0, count=1)
                Node(id=4, dist=2.0, count=2)
                    Node(id=0, dist=1.0, count=1)
                    Node(id=1, dist=3.0, count=1)
    """).strip()
    assert str(root) == expected_str

    # Also request list of nodes.
    root, nodes = anjl.to_tree(Z, rd=True)
    assert isinstance(root, anjl.Node)
    assert isinstance(nodes, list)
    assert len(nodes) == n_nodes
    for i, node in enumerate(nodes):
        assert isinstance(node, anjl.Node)
        assert node.id == i
        if i >= n_original:
            # Internal node.
            assert node.count > 1
            assert not node.is_leaf
            assert node.left is not None
            assert isinstance(node.left, anjl.Node)
            assert node.right is not None
            assert isinstance(node.right, anjl.Node)
        else:
            assert node.count == 1
            assert node.is_leaf
            assert node.left is None
            assert node.right is None

    # Count sort.
    _, nodes = anjl.to_tree(Z, rd=True, count_sort=True)
    for node in nodes:
        if not node.is_leaf:
            assert node.left.count <= node.right.count

    # Distance sort.
    _, nodes = anjl.to_tree(Z, rd=True, distance_sort=True)
    for node in nodes:
        if not node.is_leaf:
            assert node.left.dist <= node.right.dist
