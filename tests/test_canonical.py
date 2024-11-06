import anjl
from anjl.testing import validate_nj_result, numpy_copy_options
from numpy.testing import assert_allclose
from scipy.spatial.distance import squareform  # type: ignore
import pytest


def test_example_1():
    # This is example 1 from Amelia Harrison's blog.
    # https://www.tenderisthebyte.com/blog/2022/08/31/neighbor-joining-trees/

    D, _ = anjl.data.example_1()
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

    # Check condensed.
    dist = squareform(D)
    assert dist.ndim == 1
    Zc = anjl.canonical_nj(dist)
    assert_allclose(Z, Zc)


def test_wikipedia_example():
    # This example comes from the wikipedia page on neighbour-joining.
    # https://en.wikipedia.org/wiki/Neighbor_joining#Example
    D, _ = anjl.data.wikipedia_example()
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

    # Check condensed.
    dist = squareform(D)
    assert dist.ndim == 1
    Zc = anjl.canonical_nj(dist)
    assert_allclose(Z, Zc)


@pytest.mark.parametrize("copy", numpy_copy_options)
def test_mosquitoes(copy):
    D, _ = anjl.data.mosquitoes()
    Z = anjl.canonical_nj(D, copy=copy)
    validate_nj_result(Z, D)

    # Check condensed.
    D, _ = anjl.data.mosquitoes()
    dist = squareform(D)
    assert dist.ndim == 1
    Zc = anjl.canonical_nj(dist, copy=copy)
    assert_allclose(Z, Zc)
