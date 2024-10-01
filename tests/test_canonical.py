import anjl
import numpy as np


def test_canonical_nj():
    D = np.array(
        [
            [0, 5, 9, 9, 8],
            [5, 0, 10, 10, 9],
            [9, 10, 0, 8, 7],
            [9, 10, 8, 0, 3],
            [8, 9, 7, 3, 0],
        ],
        dtype=np.float32,
    )
    Z = anjl.canonical_nj(D)

    # Check basic properties of the return value.
    assert Z is not None
    assert isinstance(Z, np.ndarray)
    assert Z.ndim == 2
    assert Z.dtype == np.float32
    n_original = D.shape[0]
    n_internal = n_original - 1
    assert Z.shape == (n_internal, 5)

    # First and second column should contain child node IDs.
    n_nodes = n_original + n_internal
    assert np.all(Z[:, 0] < n_nodes)
    assert np.all(Z[:, 1] < n_nodes)

    # Third and fourth columns should be distances to child nodes.
    assert np.all(Z[:, 2] >= 0)
    assert np.all(Z[:, 3] >= 0)

    # Final column should contain number of leaves.
    assert np.all(Z[:, 4] <= n_original)
