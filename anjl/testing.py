import numpy as np
from numpy.testing import assert_allclose
from packaging.version import Version

if Version(np.__version__) >= Version("2"):
    numpy_copy_options = [True, False, None]
else:
    numpy_copy_options = [True, False]


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
    assert np.all(Z[:, 0] < (n_nodes - 1))
    assert np.all(Z[:, 1] < (n_nodes - 1))

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
