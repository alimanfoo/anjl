import anjl
import numpy as np
from numpy.testing import assert_allclose


def validate_nj_result(Z, D):
    # Check basic properties of the return value.
    assert Z is not None
    assert isinstance(Z, np.ndarray)
    assert Z.ndim == 2
    assert Z.dtype == np.float32
    n_original = D.shape[0]
    n_internal = n_original - 1
    assert Z.shape == (n_internal, 5)

    # First and second column should contain node IDs.
    n_nodes = n_original + n_internal
    assert np.all(Z[:, 0] < n_nodes - 1)
    assert np.all(Z[:, 1] < n_nodes - 1)

    # Child node IDs should appear uniquely.
    children = Z[:, 0:2].flatten()
    children.sort()
    expected_children = np.arange(n_nodes - 1, dtype=np.float32)
    assert_allclose(children, expected_children)

    # Third and fourth columns should be distances to child nodes.
    assert np.all(Z[:, 2] >= 0)
    assert np.all(Z[:, 3] >= 0)

    # Final column should contain number of leaves.
    assert np.all(Z[:, 4] <= n_original)

    # Final row should be the root.
    assert int(Z[-1, 4]) == n_original


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
    validate_nj_result(Z, D)

    # First iteration.
    left, right, ldist, rdist, leaves = Z[0]
    # Expect nodes A and B to be joined.
    assert int(left) == 0
    assert int(right) == 1
    # Expect distances are 1 and 3 respectively.
    assert_allclose(ldist, 1)
    assert_allclose(rdist, 3)
    # Expect 2 leaves in the clade.
    assert int(leaves) == 2

    # Second iteration.
    left, right, ldist, rdist, leaves = Z[1]
    # Expect nodes C and Z to be joined.
    assert int(left) == 2
    assert int(right) == 4
    assert_allclose(ldist, 2)
    assert_allclose(rdist, 2)
    assert int(leaves) == 3

    # Third iteration (termination).
    left, right, ldist, rdist, leaves = Z[2]
    # Expect nodes D and Y to be joined.
    assert int(left) == 3
    assert int(right) == 5
    # N.B., we handle termination by placing the final
    # (root) node at the midpoint between the last two
    # children. This is equivalent to placing an edge
    # directly between the last two children.
    assert_allclose(ldist, 3.5)
    assert_allclose(rdist, 3.5)
    assert int(leaves) == 4


def test_wikipedia():
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
    validate_nj_result(Z, D)

    # First iteration.
    left, right, ldist, rdist, leaves = Z[0]
    # Expect nodes 0 and 1 to be joined.
    assert int(left) == 0
    assert int(right) == 1
    # Expect distances are 2 and 3 respectively.
    assert_allclose(ldist, 2)
    assert_allclose(rdist, 3)
    # Expect 2 leaves in the clade.
    assert int(leaves) == 2

    # Further iterations cannot be tested because there are
    # different ways the remaining nodes could be joined.
