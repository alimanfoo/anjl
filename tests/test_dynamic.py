import anjl
from anjl.testing import validate_nj_result, numpy_copy_options
from numpy.testing import assert_allclose
from scipy.spatial.distance import squareform  # type: ignore
import pytest


@pytest.mark.parametrize("parallel", [False, True])
def test_example_1(parallel):
    # This is example 1 from Amelia Harrison's blog.
    # https://www.tenderisthebyte.com/blog/2022/08/31/neighbor-joining-trees/

    D, _ = anjl.data.example_1()
    Z = anjl.dynamic_nj(D, parallel=parallel)
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

    # Further iterations cannot be tested because there are
    # different ways the remaining nodes could be joined.

    # Check condensed.
    dist = squareform(D)
    assert dist.ndim == 1
    Zc = anjl.dynamic_nj(dist, parallel=parallel)
    assert_allclose(Z, Zc)


@pytest.mark.parametrize("parallel", [False, True])
def test_wikipedia_example(parallel):
    # This example comes from the wikipedia page on neighbour-joining.
    # https://en.wikipedia.org/wiki/Neighbor_joining#Example

    D, _ = anjl.data.wikipedia_example()
    Z = anjl.dynamic_nj(D, parallel=parallel)
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
    Zc = anjl.dynamic_nj(dist, parallel=parallel)
    assert_allclose(Z, Zc)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("copy", numpy_copy_options)
def test_mosquitoes(copy, parallel):
    D, _ = anjl.data.mosquitoes()
    Z = anjl.dynamic_nj(D, copy=copy, parallel=parallel)
    validate_nj_result(Z, D)

    # Check condensed.
    D, _ = anjl.data.mosquitoes()
    dist = squareform(D)
    assert dist.ndim == 1
    Zc = anjl.dynamic_nj(dist, copy=copy, parallel=parallel)
    assert_allclose(Z, Zc)
