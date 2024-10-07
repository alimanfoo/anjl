from typing import Callable
from collections.abc import Mapping
import numpy as np
import numba


def rapid_nj(
    D: np.ndarray,
    disallow_negative_distances: bool = True,
    progress: Callable | None = None,
    progress_options: Mapping = {},
) -> np.ndarray:
    """TODO"""

    # Make a copy of distance matrix D because we will overwrite it during the
    # algorithm.
    D = np.array(D, copy=True, order="C", dtype=np.float32)

    # Initialize the "divergence" array, containing sum of distances to other nodes.
    U = np.sum(D, axis=1)
    u_max = U.max()

    # Obtain node identifiers to sort the distance matrix row-wise.
    nodes_sorted = np.argsort(D, axis=1)
    assert D.shape == nodes_sorted.shape

    # Number of original observations.
    n_original = D.shape[0]

    # Expected number of new (internal) nodes that will be created.
    n_internal = n_original - 1

    # Total number of nodes in the tree, including internal nodes.
    n_nodes = n_original + n_internal

    # Map row indices to node IDs.
    index_to_id = np.arange(n_original)

    # Map node IDs to row indices.
    id_to_index = np.full(shape=n_nodes, fill_value=-1)
    id_to_index[:n_original] = np.arange(n_original)

    # Initialise output. This is similar to the output that scipy hierarchical
    # clustering functions return, where each row contains data for one internal node
    # in the tree, except that each row here contains:
    # - left child node ID
    # - right child node ID
    # - distance to left child node
    # - distance to right child node
    # - total number of leaves
    Z = np.zeros(shape=(n_internal, 5), dtype=np.float32)

    # Keep track of which nodes have been clustered and are now "obsolete". N.B., this
    # is different from canonical implementation because we index here by node ID.
    clustered = np.zeros(shape=n_nodes - 1, dtype="bool")

    # Support wrapping the iterator in a progress bar.
    iterator = range(n_internal)
    if progress:
        iterator = progress(iterator, **progress_options)

    # Begin iterating.
    for iteration in iterator:
        # Perform one iteration of the neighbour-joining algorithm.
        u_max = _rapid_nj_iteration(
            iteration=iteration,
            D=D,
            U=U,
            nodes_sorted=nodes_sorted,
            index_to_id=index_to_id,
            id_to_index=id_to_index,
            clustered=clustered,
            Z=Z,
            n_original=n_original,
            disallow_negative_distances=disallow_negative_distances,
            u_max=u_max,
        )

    return Z


@numba.njit
def _rapid_nj_iteration(
    iteration: int,
    D: np.ndarray,
    U: np.ndarray,
    nodes_sorted: np.ndarray,
    index_to_id: np.ndarray,
    id_to_index: np.ndarray,
    clustered: np.ndarray,
    Z: np.ndarray,
    n_original: int,
    disallow_negative_distances: bool,
    u_max: np.float32,
) -> np.float32:
    # This will be the identifier for the new node to be created in this iteration.
    node = iteration + n_original

    # Number of nodes remaining in this iteration.
    n_remaining = n_original - iteration

    if n_remaining > 2:
        # Search for the closest pair of nodes to join.
        i_min, j_min = _rapid_nj_search(
            D=D,
            U=U,
            nodes_sorted=nodes_sorted,
            clustered=clustered,
            index_to_id=index_to_id,
            id_to_index=id_to_index,
            n=n_remaining,
            u_max=u_max,
        )
        assert i_min >= 0
        assert j_min >= 0
        assert i_min != j_min

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
        u_max = _rapid_nj_update(
            D=D,
            U=U,
            nodes_sorted=nodes_sorted,
            index_to_id=index_to_id,
            id_to_index=id_to_index,
            clustered=clustered,
            node=node,
            child_i=child_i,
            child_j=child_j,
            i_min=i_min,
            j_min=j_min,
            d_ij=d_ij,
        )

    return u_max


@numba.njit
def _rapid_nj_search(
    D: np.ndarray,
    U: np.ndarray,
    nodes_sorted: np.ndarray,
    clustered: np.ndarray,
    index_to_id: np.ndarray,
    id_to_index: np.ndarray,
    n: int,
    u_max: np.float32,
) -> tuple[int, int]:
    # Initialize working variables.
    q_min = np.inf
    i_min = -1
    j_min = -1

    # First pass, initialise q_min with the first value in each row, which should be a
    # good candidate for the minimum value because each row is sorted.
    for i in range(nodes_sorted.shape[0]):
        # Obtain node identifier for the current row.
        id_i = index_to_id[i]
        assert id_i >= 0

        # Skip if this node is already clustered.
        if clustered[id_i]:
            continue

        # Obtain divergence for node corresponding to this row.
        u_i = U[i]

        # Search the row to find the first non-clustered value.
        for sj in range(nodes_sorted.shape[1]):
            # Obtain node identifier for the current item.
            id_j = nodes_sorted[i, sj]
            assert id_j >= 0

            # Skip if this node is already clustered or we are comparing to self.
            if id_i == id_j or clustered[id_j]:
                continue

            # Obtain column index in the distance matrix.
            j = id_to_index[id_j]
            assert j >= 0

            # Calculate q.
            d = D[i, j]
            u_j = U[j]
            q = (n - 2) * d - u_i - u_j

            # Compare with current minimum.
            if q < q_min:
                q_min = q
                i_min = i
                j_min = j

            # Break here as we only want to find the first non-clustered value in this
            # pass.
            break

    # Second pass, search all values up to threshold.
    for i in range(nodes_sorted.shape[0]):
        # Obtain node identifier for the current row.
        id_i = index_to_id[i]
        assert id_i >= 0

        # Skip if this node is already clustered.
        if clustered[id_i]:
            continue

        # Obtain divergence for node corresponding to this row.
        u_i = U[i]

        # Search the row up to threshold.
        for sj in range(nodes_sorted.shape[1]):
            # Obtain node identifier for the current item.
            id_j = nodes_sorted[i, sj]
            assert id_j >= 0

            # Skip if this node is already clustered or we are comparing to self.
            if id_i == id_j or clustered[id_j]:
                continue

            # Obtain column index in the distance matrix.
            j = id_to_index[id_j]
            assert j >= 0

            # Partially calculate q.
            d = D[i, j]
            q_partial = (n - 2) * d - u_i

            # Limit search. Because the row is sorted, if we are already above this
            # threshold then we know there is no need to search remaining nodes in the
            # row.
            if q_partial - u_max >= q_min:
                break

            # Fully calculate q.
            u_j = U[j]
            q = q_partial - u_j
            if q < q_min:
                q_min = q
                i_min = i
                j_min = j

    return i_min, j_min


@numba.njit
def _rapid_nj_update(
    D: np.ndarray,
    U: np.ndarray,
    nodes_sorted: np.ndarray,
    index_to_id: np.ndarray,
    id_to_index: np.ndarray,
    clustered: np.ndarray,
    node: int,
    child_i: int,
    child_j: int,
    i_min: int,
    j_min: int,
    d_ij: float,
) -> np.float32:
    # Update data structures. Here we obsolete the row and column corresponding to the
    # node at j_min, and we reuse the row and column at i_min for the new node.
    clustered[child_i] = True
    clustered[child_j] = True
    index_to_id[i_min] = node
    id_to_index[child_i] = -1
    id_to_index[child_j] = -1
    id_to_index[node] = i_min

    # Subtract out the distances for the nodes that have just been joined.
    U -= D[i_min]
    U -= D[j_min]
    U[j_min] = 0  # Set 0 to make sure max calculation is correct.

    # Initialize divergence for the new node.
    u_new = np.float32(0)

    # Find new max.
    u_max = np.float32(0)

    # Update distances and divergence.
    for k in range(D.shape[0]):
        id_k = index_to_id[k]

        if clustered[id_k]:
            D[i_min, k] = np.inf
            D[k, i_min] = np.inf
            continue

        if k == i_min or k == j_min:
            continue

        # Distance from k to the new node.
        d_ik = D[i_min, k]
        d_jk = D[j_min, k]
        d_k = 0.5 * (d_ik + d_jk - d_ij)
        D[i_min, k] = d_k
        D[k, i_min] = d_k
        u_k = U[k] + d_k
        U[k] = u_k

        # Record new max.
        if u_k > u_max:
            u_max = u_k

        # Accumulate divergence for the new node.
        u_new += d_k

    # Store divergence for the new node.
    U[i_min] = u_new

    # Record new max.
    if u_new > u_max:
        u_max = u_new

    # Update the sorted distances and indices for the new node.
    distances_new = D[i_min]
    indices_new = np.argsort(distances_new)
    ids_new = np.take(index_to_id, indices_new)
    nodes_sorted[i_min] = ids_new

    return u_max
