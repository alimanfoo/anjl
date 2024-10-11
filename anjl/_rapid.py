import numpy as np
from numpy.typing import NDArray
from numba import njit, int64, float32, bool_
from numpydoc_decorator import doc
from . import params


INT64_MIN = np.int64(np.iinfo(np.int64).min)
FLOAT32_INF = np.float32(np.inf)
NOGIL = True
FASTMATH = False  # setting True actually seems to slow things down
ERROR_MODEL = "numpy"
BOUNDSCHECK = False


@doc(
    summary="""Perform neighbour-joining using an algorithm based on Simonsen et al. [1]_""",
    extended_summary="""
        This implementation builds and maintains a sorted copy of the distance matrix
        and uses heuristics to avoid searching pairs that cannot possibly be neighbours
        in each iteration. In the worst case it has complexity O(n^3) like the canonical
        algorithm but in practice it usually scales closer to O(n^2).
    """,
    notes="""
        The ordering of the internal nodes may be different between the canonical and
        the rapid algorithms, because these algorithms search the distance matrix in a
        different order. However, the resulting trees will be topologically equivalent.
    """,
    references={
        "1": "https://pure.au.dk/ws/files/19821675/rapidNJ.pdf",
    },
)
def rapid_nj(
    D: params.D,
    disallow_negative_distances: params.disallow_negative_distances = True,
    progress: params.progress = None,
    progress_options: params.progress_options = {},
    gc: params.gc = 100,
) -> params.Z:
    # Make a copy of distance matrix D because we will overwrite it during the
    # algorithm.
    D_copy: NDArray[np.float32] = np.array(D, copy=True, order="C", dtype=np.float32)
    del D

    # Initialize the "divergence" array, containing sum of distances to other nodes.
    U: NDArray[np.float32] = np.sum(D_copy, axis=1, dtype=np.float32)
    u_max = U.max()

    # Set up a sorted version of the distance array.
    D_sorted, nodes_sorted = rapid_setup_distance(D_copy)

    # Number of original observations.
    n_original = D_copy.shape[0]

    # Expected number of new (internal) nodes that will be created.
    n_internal = n_original - 1

    # Total number of nodes in the tree, including internal nodes.
    n_nodes = n_original + n_internal

    # Map row indices to node IDs.
    index_to_id: NDArray[np.int64] = np.arange(n_original, dtype=np.int64)

    # Map node IDs to row indices.
    id_to_index: NDArray[np.int64] = np.full(
        shape=n_nodes, dtype=np.int64, fill_value=INT64_MIN
    )
    id_to_index[:n_original] = np.arange(n_original)

    # Initialise output. This is similar to the output that scipy hierarchical
    # clustering functions return, where each row contains data for one internal node
    # in the tree, except that each row here contains:
    # - left child node ID
    # - right child node ID
    # - distance to left child node
    # - distance to right child node
    # - total number of leaves
    Z: NDArray[np.float32] = np.zeros(shape=(n_internal, 5), dtype=np.float32)

    # Keep track of which nodes have been clustered and are now "obsolete". N.B., this
    # is different from canonical implementation because we index here by node ID.
    clustered: NDArray[np.bool_] = np.zeros(shape=n_nodes - 1, dtype=np.bool_)

    # Convenience to also keep track of which rows are no longer in use.
    obsolete: NDArray[np.bool_] = np.zeros(shape=n_original, dtype=np.bool_)

    # Support wrapping the iterator in a progress bar like tqdm.
    iterator = range(n_internal)
    if progress:
        iterator = progress(iterator, **progress_options)

    # Begin iterating.
    for iteration in iterator:
        # Number of nodes remaining in this iteration.
        n_remaining = n_original - iteration

        # Garbage collection.
        if gc and iteration > 0 and iteration % gc == 0:
            nodes_sorted, D_sorted = rapid_gc(
                nodes_sorted=nodes_sorted,
                D_sorted=D_sorted,
                clustered=clustered,
                obsolete=obsolete,
                n_remaining=n_remaining,
            )

        # Perform one iteration of the neighbour-joining algorithm.
        u_max = rapid_iteration(
            iteration=iteration,
            D=D_copy,
            D_sorted=D_sorted,
            U=U,
            nodes_sorted=nodes_sorted,
            index_to_id=index_to_id,
            id_to_index=id_to_index,
            clustered=clustered,
            obsolete=obsolete,
            Z=Z,
            n_original=n_original,
            disallow_negative_distances=disallow_negative_distances,
            u_max=u_max,
        )

    return Z


@njit(
    (float32[:, :],),
    nogil=NOGIL,
    fastmath=FASTMATH,
    error_model=ERROR_MODEL,
    boundscheck=BOUNDSCHECK,
)
def rapid_setup_distance(D: NDArray[np.float32]):
    # Set the diagonal and upper triangle to inf so we can skip self-comparisons and
    # avoid double-comparison between leaf nodes.
    D_sorted = np.full(shape=D.shape, dtype=float32, fill_value=FLOAT32_INF)
    nodes_sorted = np.full(shape=D.shape, dtype=int64, fill_value=INT64_MIN)
    for i in range(D.shape[0]):
        D[i, i] = FLOAT32_INF  # avoid self comparisons in all iterations
        d = D[i, :i]
        nx = np.argsort(d)
        dx = d[nx]
        D_sorted[i, :i] = dx
        nodes_sorted[i, :i] = nx
    return D_sorted, nodes_sorted


@njit(
    (
        float32[:, :],  # D_sorted
        int64[:, :],  # nodes_sorted
        bool_[:],  # clustered
        bool_[:],  # obsolete
        int64,  # n_remaining
    ),
    nogil=NOGIL,
    fastmath=FASTMATH,
    error_model=ERROR_MODEL,
    boundscheck=BOUNDSCHECK,
)
def rapid_gc(
    D_sorted: NDArray[np.float32],
    nodes_sorted: NDArray[np.int64],
    clustered: NDArray[np.bool_],
    obsolete: NDArray[np.bool_],
    n_remaining: int,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    for i in range(nodes_sorted.shape[0]):
        if obsolete[i]:
            continue
        j_new = 0
        for j in range(nodes_sorted.shape[1]):
            node_j = nodes_sorted[i, j]
            if node_j < 0:
                break
            if clustered[node_j]:
                continue
            nodes_sorted[i, j_new] = node_j
            D_sorted[i, j_new] = D_sorted[i, j]
            j_new += 1
    nodes_sorted = nodes_sorted[:, :n_remaining]
    D_sorted = D_sorted[:, :n_remaining]
    return nodes_sorted, D_sorted


@njit(
    (
        float32[:, :],  # D_sorted
        float32[:],  # U
        int64[:, :],  # nodes_sorted
        bool_[:],  # clustered
        bool_[:],  # obsolete
        int64[:],  # id_to_index
        int64,  # n_remaining
        float32,  # u_max
    ),
    nogil=NOGIL,
    fastmath=FASTMATH,
    error_model=ERROR_MODEL,
    boundscheck=BOUNDSCHECK,
)
def rapid_search(
    D_sorted: NDArray[np.float32],
    U: NDArray[np.float32],
    nodes_sorted: NDArray[np.int64],
    clustered: NDArray[np.bool_],
    obsolete: NDArray[np.bool_],
    id_to_index: NDArray[np.int64],
    n_remaining: int,
    u_max: np.float32,
) -> tuple[np.int64, np.int64]:
    # Initialize working variables.
    q_min = FLOAT32_INF
    threshold = FLOAT32_INF
    i_min = INT64_MIN
    j_min = INT64_MIN
    coefficient = np.float32(n_remaining - 2)
    m = nodes_sorted.shape[0]
    n = nodes_sorted.shape[1]
    assert m == D_sorted.shape[0]
    assert n == D_sorted.shape[1]

    # Search all values up to threshold.
    for i in range(m):
        # Skip if row is no longer in use.
        if obsolete[i]:
            continue

        # Obtain divergence for node corresponding to this row.
        u_i = U[i]

        # Search the row up to threshold.
        for s in range(0, n):
            # Obtain node identifier for the current item.
            node_j = nodes_sorted[i, s]

            # Break at end of nodes.
            if node_j < 0:
                break

            # Skip if this node is already clustered.
            if clustered[node_j]:
                continue

            # Access distance.
            d = D_sorted[i, s]

            # Partially calculate q.
            q_partial = coefficient * d - u_i

            # Limit search. Because the row is sorted, if we are already above this
            # threshold then we know there is no need to search remaining nodes in the
            # row.
            if q_partial >= threshold:
                break

            # Fully calculate q.
            j = id_to_index[node_j]
            u_j = U[j]
            q = q_partial - u_j

            if q < q_min:
                q_min = q
                threshold = q_min + u_max
                i_min = np.int64(i)
                j_min = np.int64(j)

    return i_min, j_min


@njit(
    (
        float32[:, :],  # D
        float32[:, :],  # D_sorted
        float32[:],  # U
        int64[:, :],  # nodes_sorted
        int64[:],  # index_to_id
        int64[:],  # id_to_index
        bool_[:],  # clustered
        bool_[:],  # obsolete
        int64,  # parent
        int64,  # child_i
        int64,  # child_j
        int64,  # i_min
        int64,  # j_min
        float32,  # d_ij
    ),
    nogil=NOGIL,
    fastmath=FASTMATH,
    error_model=ERROR_MODEL,
    boundscheck=BOUNDSCHECK,
)
def rapid_update(
    D: NDArray[np.float32],
    D_sorted: NDArray[np.float32],
    U: NDArray[np.float32],
    nodes_sorted: NDArray[np.int64],
    index_to_id: NDArray[np.int64],
    id_to_index: NDArray[np.int64],
    clustered: NDArray[np.bool_],
    obsolete: NDArray[np.bool_],
    parent: np.int64,
    child_i: np.int64,
    child_j: np.int64,
    i_min: np.int64,
    j_min: np.int64,
    d_ij: np.float32,
) -> np.float32:
    # Update data structures. Here we obsolete the row corresponding to the node at
    # j_min, and we reuse the row at i_min for the new node.
    clustered[child_i] = True
    clustered[child_j] = True

    # Assign the new node to row at i_min.
    index_to_id[i_min] = parent
    id_to_index[parent] = i_min

    # Obsolete the row of data corresponding to the node at j_min.
    obsolete[j_min] = True

    # Initialize divergence for the new node.
    u_new = np.float32(0)

    # Find new max.
    u_max = np.float32(0)

    # Update distances and divergence.
    for k in range(D.shape[0]):
        if k == i_min or k == j_min or obsolete[k]:
            continue

        # Calculate distance from k to the new node.
        d_ki = D[k, i_min]
        d_kj = D[k, j_min]
        d_k_new = 0.5 * (d_ki + d_kj - d_ij)
        D[i_min, k] = d_k_new
        D[k, i_min] = d_k_new

        # Subtract out the distances for the nodes that have just been joined and add
        # in distance for the new node.
        u_k = U[k] - d_ki - d_kj + d_k_new
        U[k] = u_k

        # Record new max.
        if u_k > u_max:
            u_max = u_k

        # Accumulate divergence for the new node.
        u_new += d_k_new

        # Distance from k to the obsolete node.
        # D[j_min, k] = FLOAT32_INF  # not needed as this row is obsolete and never read
        D[k, j_min] = FLOAT32_INF

    # Store divergence for the new node.
    U[i_min] = u_new

    # Record new max.
    if u_new > u_max:
        u_max = u_new

    # First cut down to just the active nodes.
    active = ~obsolete
    distances_new = D[i_min, active]
    nodes_active = index_to_id[active]

    # Now sort the new distances.
    indices_sorted = np.argsort(distances_new)
    nodes_sorted_new = nodes_active[indices_sorted]
    distances_sorted_new = distances_new[indices_sorted]

    # Now update sorted nodes and distances.
    p = nodes_sorted_new.shape[0]
    nodes_sorted[i_min, :p] = nodes_sorted_new
    nodes_sorted[i_min, p:] = INT64_MIN
    D_sorted[i_min, :p] = distances_sorted_new
    D_sorted[i_min, p:] = FLOAT32_INF

    return u_max


@njit(
    (
        int64,  # iteration
        float32[:, :],  # D
        float32[:, :],  # D_sorted
        float32[:],  # U
        int64[:, :],  # nodes_sorted
        int64[:],  # index_to_id
        int64[:],  # id_to_index
        bool_[:],  # clustered
        bool_[:],  # obsolete
        float32[:, :],  # Z
        int64,  # n_original
        bool_,  # disallow_negative_distances
        float32,  # u_max
    ),
    nogil=NOGIL,
    fastmath=FASTMATH,
    error_model=ERROR_MODEL,
    boundscheck=BOUNDSCHECK,
)
def rapid_iteration(
    iteration: int,
    D: NDArray[np.float32],
    D_sorted: NDArray[np.float32],
    U: NDArray[np.float32],
    nodes_sorted: NDArray[np.int64],
    index_to_id: NDArray[np.int64],
    id_to_index: NDArray[np.int64],
    clustered: NDArray[np.bool_],
    obsolete: NDArray[np.bool_],
    Z: NDArray[np.float32],
    n_original: int,
    disallow_negative_distances: bool,
    u_max: np.float32,
) -> np.float32:
    # This will be the identifier for the new node to be created in this iteration.
    parent = iteration + n_original

    # Number of nodes remaining in this iteration.
    n_remaining = n_original - iteration

    if n_remaining > 2:
        # Search for the closest pair of nodes to join.
        i_min, j_min = rapid_search(
            D_sorted=D_sorted,
            U=U,
            nodes_sorted=nodes_sorted,
            clustered=clustered,
            obsolete=obsolete,
            id_to_index=id_to_index,
            n_remaining=n_remaining,
            u_max=u_max,
        )

        # Get IDs for the nodes to be joined.
        child_i = index_to_id[i_min]
        child_j = index_to_id[j_min]

        # Calculate distances to the new internal node.
        d_ij = D[i_min, j_min]
        d_i = 0.5 * (d_ij + (1 / (n_remaining - 2)) * (U[i_min] - U[j_min]))
        d_j = 0.5 * (d_ij + (1 / (n_remaining - 2)) * (U[j_min] - U[i_min]))

    else:
        # Termination. Join the two remaining nodes, placing the final node at the
        # midpoint.
        child_i, child_j = np.nonzero(~clustered)[0]
        i_min = id_to_index[child_i]
        j_min = id_to_index[child_j]
        d_ij = D[i_min, j_min]
        d_i = d_ij / 2
        d_j = d_ij / 2

    # Sanity checks.
    assert i_min >= 0
    assert j_min >= 0
    assert i_min != j_min
    assert child_i >= 0
    assert child_j >= 0
    assert child_i != child_j

    # Handle possibility of negative distances.
    if disallow_negative_distances:
        d_i = max(0, d_i)
        d_j = max(0, d_j)

    # Stabilise ordering for easier comparisons.
    if child_i > child_j:
        child_i, child_j = child_j, child_i
        i_min, j_min = j_min, i_min
        d_i, d_j = d_j, d_i

    # Get number of leaves.
    if child_i < n_original:
        leaves_i = 1
    else:
        leaves_i = Z[child_i - n_original, 4]
    if child_j < n_original:
        leaves_j = 1
    else:
        leaves_j = Z[child_j - n_original, 4]

    # Store new node data.
    Z[iteration, 0] = child_i
    Z[iteration, 1] = child_j
    Z[iteration, 2] = d_i
    Z[iteration, 3] = d_j
    Z[iteration, 4] = leaves_i + leaves_j

    if n_remaining > 2:
        # Update data structures.
        u_max = rapid_update(
            D=D,
            D_sorted=D_sorted,
            U=U,
            nodes_sorted=nodes_sorted,
            index_to_id=index_to_id,
            id_to_index=id_to_index,
            clustered=clustered,
            obsolete=obsolete,
            parent=parent,
            child_i=child_i,
            child_j=child_j,
            i_min=i_min,
            j_min=j_min,
            d_ij=d_ij,
        )

    return u_max
