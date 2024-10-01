import numpy as np
import numba  # type: ignore


def canonical_nj(
    D,
    disallow_negative_distances=True,
    progress=None,
    progress_options={},
):
    """TODO"""

    # Make a copy of distance matrix D because we will overwrite it during
    # the algorithm.
    D = np.array(D, copy=True, order="C", dtype=np.float32)

    # Number of original observations.
    n_original = D.shape[0]

    # Expected number of new (internal) nodes that will be created.
    n_internal = n_original - 1

    # Map row indices to node IDs.
    index_to_id = np.arange(n_original)

    # Initialise output. This is similar to the output that scipy
    # hierarchical clustering functions return, where each row contains
    # data for one internal node in the tree, except that each row here
    # contains:
    # - left child node ID
    # - right child node ID
    # - distance to left child node
    # - distance to right child node
    # - total number of leaves
    Z = np.zeros(shape=(n_internal, 5), dtype=np.float32)

    # Initialize the "divergence" array, containing sum of distances
    # to other nodes.
    U = np.sum(D, axis=1)

    # Keep track of which rows correspond to nodes that have been clustered
    # and are now "obsolete".
    obsolete = np.zeros(shape=n_original, dtype="bool")

    # Support wrapping the iterator in a progress bar.
    iterator = range(n_internal)
    if progress:
        iterator = progress(iterator, **progress_options)

    # Begin iterating.
    for iteration in iterator:
        # Perform one iteration of the neighbour-joining algorithm.
        _canonical_nj_iteration(
            iteration=iteration,
            D=D,
            U=U,
            index_to_id=index_to_id,
            obsolete=obsolete,
            Z=Z,
            n_original=n_original,
            disallow_negative_distances=disallow_negative_distances,
        )

    return Z


@numba.njit
def _canonical_nj_iteration(
    iteration,
    D,
    U,
    index_to_id,
    obsolete,
    Z,
    n_original,
    disallow_negative_distances,
):
    # This will be the identifier for the new node to be created in this
    # iteration.
    node_id = iteration + n_original

    # Number of nodes remaining in this iteration.
    n_remaining = n_original - iteration

    if n_remaining > 2:
        # Search for the closest pair of nodes to join.
        i_min, j_min = _canonical_nj_search(D=D, U=U, obsolete=obsolete, n=n_remaining)
        assert i_min >= 0
        assert j_min >= 0
        assert i_min != j_min

        # Calculate distances to the new internal node.
        d_ij = D[i_min, j_min]
        d_i = 0.5 * (d_ij + (1 / (n_remaining - 2)) * (U[i_min] - U[j_min]))
        d_j = 0.5 * (d_ij + (1 / (n_remaining - 2)) * (U[j_min] - U[i_min]))

    else:
        # Termination. Join the two remaining nodes, placing the final node at the
        # midpoint.
        i_min, j_min = np.nonzero(~obsolete)[0]
        d_ij = D[i_min, j_min]
        d_i = d_ij / 2
        d_j = d_ij / 2

    # Handle possibility of negative distances.
    if disallow_negative_distances:
        d_i = max(0, d_i)
        d_j = max(0, d_j)

    # Get IDs for the nodes to be joined.
    id_i = index_to_id[i_min]
    id_j = index_to_id[j_min]
    assert id_i >= 0
    assert id_j >= 0
    assert id_i != id_j

    # Get number of leaves.
    if id_i < n_original:
        leaves_i = 1
    else:
        leaves_i = Z[id_i - n_original, 4]
    if id_j < n_original:
        leaves_j = 1
    else:
        leaves_j = Z[id_j - n_original, 4]

    # Store new node data.
    Z[iteration, 0] = id_i
    Z[iteration, 1] = id_j
    Z[iteration, 2] = d_i
    Z[iteration, 3] = d_j
    Z[iteration, 4] = leaves_i + leaves_j

    if n_remaining > 2:
        # Update data structures.
        _canonical_nj_update(
            D=D,
            U=U,
            index_to_id=index_to_id,
            obsolete=obsolete,
            node_id=node_id,
            i_min=i_min,
            j_min=j_min,
        )


@numba.njit
def _canonical_nj_search(D, U, obsolete, n):
    # Search for the closest pair of neighbouring nodes to join.
    q_min = np.inf
    i_min = -1
    j_min = -1
    for i in range(D.shape[0]):
        if obsolete[i]:
            continue
        u_i = U[i]
        for j in range(i):
            if obsolete[j]:
                continue
            u_j = U[j]
            d = D[i, j]
            q = (n - 2) * d - u_i - u_j
            if q < q_min:
                q_min = q
                i_min = i
                j_min = j
    return i_min, j_min


@numba.njit
def _canonical_nj_update(D, U, index_to_id, obsolete, node_id, i_min, j_min):
    # Here we obsolete the row and column corresponding to the node at j_min,
    # and we reuse the row and column at i_min for the new node.
    obsolete[j_min] = True
    index_to_id[i_min] = node_id
    index_to_id[j_min] = -1

    # Distance between nodes being joined.
    d_ij = D[i_min, j_min]

    # Subtract out the distances for the nodes that have just been joined.
    U -= D[i_min]
    U -= D[j_min]

    # Initialize divergence for the new node.
    u_new = np.float32(0)

    # Update distances and divergence.
    for k in range(D.shape[0]):
        if obsolete[k] or k == i_min:
            continue
        d_ik = D[i_min, k]
        d_jk = D[j_min, k]

        # Distance from k to the new node.
        d_k = 0.5 * (d_ik + d_jk - d_ij)
        D[i_min, k] = d_k
        D[k, i_min] = d_k
        U[k] += d_k

        # Accumulate divergence for the new node.
        u_new += d_k

    # Assign divergence for the new node.
    U[i_min] = u_new