import numpy as np
from numpy.typing import NDArray
from numba import njit, uintp, float32, bool_, void, prange, get_num_threads
from numpydoc_decorator import doc
from . import params
from ._util import (
    NUMBA_NOGIL,
    NUMBA_FASTMATH,
    NUMBA_ERROR_MODEL,
    NUMBA_BOUNDSCHECK,
    NUMBA_CACHE,
    FLOAT32_INF,
    UINTP_MAX,
    ensure_condensed_distance,
    setup_divergence,
    condensed_index,
    condensed_offset,
)


@doc(
    summary="""Perform neighbour-joining using the canonical algorithm.""",
    extended_summary="""
        This implementation performs a full scan of the distance matrix in each
        iteration of the algorithm to find the pair of nearest neighbours. It is
        therefore slower and scales with the cube of the number of original observations
        in the distance matrix, i.e., O(n^3).
    """,
)
def canonical_nj(
    D: params.D,
    disallow_negative_distances: params.disallow_negative_distances = True,
    progress: params.progress = None,
    progress_options: params.progress_options = {},
    copy: params.copy = True,
    parallel: params.parallel = True,
) -> params.Z:
    # Set up the distance matrix, ensure it is in condensed form.
    distance, n_original = ensure_condensed_distance(D=D, copy=copy)
    del D

    # Expected number of new (internal) nodes that will be created.
    n_internal = n_original - 1

    # Map row indices to node IDs.
    index_to_id = np.arange(n_original, dtype=np.uintp)

    # Initialise output. This is similar to the output that scipy hierarchical
    # clustering functions return, where each row contains data for one internal node
    # in the tree, except that each row here contains:
    # - left child node ID
    # - right child node ID
    # - distance to left child node
    # - distance to right child node
    # - total number of leaves
    Z = np.zeros(shape=(n_internal, 5), order="C", dtype=np.float32)

    # Initialize the "divergence" array, containing sum of distances to other nodes.
    R = setup_divergence(distance=distance, n_original=n_original)

    # Keep track of which rows correspond to nodes that have been clustered.
    obsolete = np.zeros(shape=n_original, dtype=np.bool_)

    # Support wrapping the iterator in a progress bar.
    iterator = range(n_internal)
    if progress:
        iterator = progress(iterator, **progress_options)

    # Begin iterating.
    for iteration in iterator:
        # Perform one iteration of the neighbour-joining algorithm.
        canonical_iteration(
            iteration=np.uintp(iteration),
            distance=distance,
            R=R,
            index_to_id=index_to_id,
            obsolete=obsolete,
            Z=Z,
            n_original=np.uintp(n_original),
            disallow_negative_distances=disallow_negative_distances,
            parallel=parallel,
        )

    return Z


@njit(
    (
        float32[::1],  # distance
        float32[::1],  # R
        bool_[::1],  # obsolete
        uintp,  # n_remaining
        uintp,  # n_original
    ),
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
    error_model=NUMBA_ERROR_MODEL,
    boundscheck=NUMBA_BOUNDSCHECK,
    cache=NUMBA_CACHE,
)
def canonical_search(
    distance: NDArray[np.float32],
    R: NDArray[np.float32],
    obsolete: NDArray[np.bool_],
    n_remaining: np.uintp,
    n_original: np.uintp,
) -> tuple[np.uintp, np.uintp]:
    """Search for the closest pair of neighbouring nodes to join."""
    # Global minimum join criterion.
    q_xy = FLOAT32_INF

    # Indices of the pair of nodes with the global minimum, to be joined.
    x = UINTP_MAX
    y = UINTP_MAX

    # Partially compute outside loop.
    coefficient = float32(n_remaining - 2)

    # Iterate over rows of the distance matrix.
    for _i in range(n_original):
        i = np.uintp(_i)  # use unsigned int for faster indexing

        # Check if row is still in use.
        if obsolete[i]:
            continue

        # Access divergence for current row.
        r_i = R[i]

        # Compute offset into condensed distance matrix.
        _offset = condensed_offset(_i, n_original)

        # Iterate over columns of the distance matrix upper triangle.
        for _j in range(i + 1, n_original):
            j = np.uintp(_j)  # use unsigned int for faster indexing

            # Check if column is still in use.
            if obsolete[j]:
                continue

            # Access divergence for the current column.
            r_j = R[j]

            # Compute index into condensed distance matrix.
            c = np.uintp(_offset + _j)

            # Compute join criterion.
            d = distance[c]
            q = coefficient * d - r_i - r_j

            if q < q_xy:
                # Found new global minimum.
                q_xy = q
                x = i
                y = j

    return x, y


@njit(
    (
        float32[::1],  # distance
        float32[::1],  # R
        bool_[::1],  # obsolete
        uintp,  # n_remaining
        uintp,  # n_original
    ),
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
    error_model=NUMBA_ERROR_MODEL,
    boundscheck=NUMBA_BOUNDSCHECK,
    # cache=NUMBA_CACHE,  # warning that cannot cache, though not clear why
    parallel=True,
)
def canonical_search_parallel(
    distance: NDArray[np.float32],
    R: NDArray[np.float32],
    obsolete: NDArray[np.bool_],
    n_remaining: np.uintp,
    n_original: np.uintp,
) -> tuple[np.uintp, np.uintp]:
    """Search for the closest pair of neighbouring nodes to join."""
    # Partially compute outside loop.
    coefficient = float32(n_remaining - 2)

    # Number of available threads.
    n_threads = get_num_threads()

    # Arrays to store thread results.
    results_q_xy = np.empty(n_threads, dtype=np.float32)
    results_xy = np.empty((n_threads, 2), dtype=np.uintp)

    # Set up parallel threads.
    for t in prange(n_threads):
        # Thread local variables.
        local_q_xy = FLOAT32_INF
        local_x = UINTP_MAX
        local_y = UINTP_MAX

        # Iterate over rows of the distance matrix, striped work distribution.
        for _i in range(t, n_original, n_threads):
            i = np.uintp(_i)  # use unsigned int for faster indexing

            # Check if row is still in use.
            if obsolete[i]:
                continue

            # Access divergence for current row.
            r_i = R[i]

            # Compute offset into condensed distance matrix.
            _offset = condensed_offset(_i, n_original)

            # Iterate over columns of the distance matrix upper triangle.
            for _j in range(i + 1, n_original):
                j = np.uintp(_j)  # use unsigned int for faster indexing

                # Check if column is still in use.
                if obsolete[j]:
                    continue

                # Access divergence for the current column.
                r_j = R[j]

                # Compute index into condensed distance matrix.
                c = np.uintp(_offset + _j)

                # Compute join criterion.
                d = distance[c]
                q = coefficient * d - r_i - r_j

                if q < local_q_xy:
                    # Found new global minimum.
                    local_q_xy = q
                    local_x = i
                    local_y = j

        # Store results for this thread.
        results_q_xy[t] = local_q_xy
        results_xy[t, 0] = local_x
        results_xy[t, 1] = local_y

    # Final reduction across thread results.
    q_xy = FLOAT32_INF
    x = UINTP_MAX
    y = UINTP_MAX
    for t in range(n_threads):
        q = results_q_xy[t]
        if q < q_xy:
            q_xy = q
            x = results_xy[t, 0]
            y = results_xy[t, 1]

    return x, y


@njit(
    void(
        float32[::1],  # distance
        float32[::1],  # R
        uintp[::1],  # index_to_id
        bool_[::1],  # obsolete
        uintp,  # parent
        uintp,  # x
        uintp,  # y
        float32,  # d_xy
        uintp,  # n_original
    ),
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
    error_model=NUMBA_ERROR_MODEL,
    boundscheck=NUMBA_BOUNDSCHECK,
    cache=NUMBA_CACHE,
)
def canonical_update(
    distance: NDArray[np.float32],
    R: NDArray[np.float32],
    index_to_id: NDArray[np.uintp],
    obsolete: NDArray[np.bool_],
    parent: np.uintp,
    x: np.uintp,
    y: np.uintp,
    d_xy: np.float32,
    n_original: np.uintp,
) -> None:
    # Here we obsolete the row and column corresponding to the node at y, and we
    # reuse the row and column at x for the new node.
    obsolete[y] = True

    # Row index to be used for the new node.
    z = x

    # Node identifier.
    index_to_id[z] = parent

    # Initialize divergence for the new node.
    r_z = float32(0)

    # Update distances and divergence.
    for _k in range(n_original):
        k = np.uintp(_k)

        if obsolete[k] or k == x or k == y:
            continue

        # Calculate and store distance from k to the new node.
        c_kx = condensed_index(k, x, n_original)
        d_kx = distance[c_kx]
        c_ky = condensed_index(k, y, n_original)
        d_ky = distance[c_ky]
        d_kz = float32(0.5) * (d_kx + d_ky - d_xy)
        c_kz = c_kx
        distance[c_kz] = d_kz

        # Subtract out the distances for the nodes that have just been joined and add
        # in distance for the new node.
        r_k = R[k] - d_kx - d_ky + d_kz
        R[k] = r_k

        # Accumulate divergence for the new node.
        r_z += d_kz

    # Assign divergence for the new node.
    R[z] = r_z


@njit(
    void(
        uintp,  # iteration
        float32[::1],  # distance
        float32[::1],  # R
        uintp[::1],  # index_to_id
        bool_[::1],  # obsolete
        float32[:, ::1],  # Z
        uintp,  # n_original
        bool_,  # disallow_negative_distances
        bool_,  # parallel
    ),
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
    error_model=NUMBA_ERROR_MODEL,
    boundscheck=NUMBA_BOUNDSCHECK,
    # cache=NUMBA_CACHE,  # warning that cannot cache, though not clear why
)
def canonical_iteration(
    iteration: np.uintp,
    distance: NDArray[np.float32],
    R: NDArray[np.float32],
    index_to_id: NDArray[np.uintp],
    obsolete: NDArray[np.bool_],
    Z: NDArray[np.float32],
    n_original: np.uintp,
    disallow_negative_distances: bool,
    parallel: bool,
) -> None:
    # This will be the identifier for the new node to be created in this iteration.
    parent = iteration + n_original

    # Number of nodes remaining in this iteration.
    n_remaining = n_original - iteration

    if n_remaining > 2:
        # Search for the closest pair of nodes to join.
        if parallel:
            x, y = canonical_search_parallel(
                distance=distance,
                R=R,
                obsolete=obsolete,
                n_remaining=n_remaining,
                n_original=n_original,
            )
        else:
            x, y = canonical_search(
                distance=distance,
                R=R,
                obsolete=obsolete,
                n_remaining=n_remaining,
                n_original=n_original,
            )
        # TODO return d_xy

        # Calculate distances to the new internal node.
        c_xy = condensed_index(x, y, n_original)
        d_xy = distance[c_xy]
        d_xz = 0.5 * (d_xy + (1 / (n_remaining - 2)) * (R[x] - R[y]))
        d_yz = 0.5 * (d_xy + (1 / (n_remaining - 2)) * (R[y] - R[x]))

    else:
        # Termination. Join the two remaining nodes, placing the final node at the
        # midpoint.
        _x, _y = np.nonzero(~obsolete)[0]
        x = np.uintp(_x)
        y = np.uintp(_y)
        c_xy = condensed_index(x, y, n_original)
        d_xy = distance[c_xy]
        d_xz = d_xy / 2
        d_yz = d_xy / 2

    # Handle possibility of negative distances.
    if disallow_negative_distances:
        d_xz = max(float32(0), d_xz)
        d_yz = max(float32(0), d_yz)

    # Get IDs for the nodes to be joined.
    child_x = index_to_id[x]
    child_y = index_to_id[y]

    # Sanity checks.
    assert x >= 0
    assert y >= 0
    assert x < n_original
    assert y < n_original
    assert x != y
    assert child_x >= 0
    assert child_y >= 0
    assert child_x != child_y

    # Stabilise ordering for easier comparisons.
    if child_x > child_y:
        child_x, child_y = child_y, child_x
        x, y = y, x
        d_xz, d_yz = d_yz, d_xz

    # Get number of leaves.
    if child_x < n_original:
        leaves_x = float32(1)
    else:
        leaves_x = Z[child_x - n_original, 4]
    if child_y < n_original:
        leaves_y = float32(1)
    else:
        leaves_y = Z[child_y - n_original, 4]

    # Store new node data.
    Z[iteration, 0] = child_x
    Z[iteration, 1] = child_y
    Z[iteration, 2] = d_xz
    Z[iteration, 3] = d_yz
    Z[iteration, 4] = leaves_x + leaves_y

    if n_remaining > 2:
        # Update data structures.
        canonical_update(
            distance=distance,
            R=R,
            index_to_id=index_to_id,
            obsolete=obsolete,
            parent=parent,
            x=x,
            y=y,
            d_xy=d_xy,
            n_original=n_original,
        )
