from typing import Callable
from collections.abc import Mapping
import numpy as np
import numba
# import time


@numba.njit
def _setup_distance(D):
    # Set the diagonal and upper triangle to inf so we can skip self-comparisons and
    # avoid double-comparison between leaf nodes.
    D_sorted = D.copy()
    for i in range(D_sorted.shape[0]):
        for j in range(i, D_sorted.shape[1]):
            D_sorted[i, j] = np.inf
    return D_sorted


def rapid_nj(
    D: np.ndarray,
    disallow_negative_distances: bool = True,
    progress: Callable | None = None,
    progress_options: Mapping = {},
    # diagnostics=False,
    gc=100,
    # ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
) -> np.ndarray:
    """TODO"""

    # Make a copy of distance matrix D because we will overwrite it during the
    # algorithm.
    D = np.array(D, copy=True, order="C", dtype=np.float32)

    # Initialize the "divergence" array, containing sum of distances to other nodes.
    U = np.sum(D, axis=1, dtype=np.float32)
    u_max = U.max()

    # Set diagonal to inf to avoid self comparison sorting first.
    # for i in range(D.shape[0]):
    #     D[i, i] = np.inf
    #
    # Obtain node identifiers to sort the distance matrix row-wise.
    # nodes_sorted = np.argsort(D, axis=1)
    #
    # Make another copy of the distance matrix sorted.
    # D_sorted = np.take_along_axis(D, nodes_sorted, axis=1)

    # Set up a sorted version of the distance array.
    D_sorted = _setup_distance(D)
    nodes_sorted = np.argsort(D_sorted, axis=1)
    D_sorted = np.take_along_axis(D_sorted, nodes_sorted, axis=1)

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
    clustered = np.zeros(shape=n_nodes - 1, dtype=bool)

    # Convenience to also keep track of which rows are no longer in use.
    obsolete = np.zeros(shape=n_original, dtype=bool)

    # Support wrapping the iterator in a progress bar.
    iterator = range(n_internal)
    if progress:
        iterator = progress(iterator, **progress_options)

    # Record iteration timings.
    # timings = []
    # searches = []
    # visits = []

    # Begin iterating.
    for iteration in iterator:
        # print("")
        # print("iteration", iteration)
        # print("D\n", D)
        # print("D_sorted\n", D_sorted)
        # print("nodes_sorted\n", nodes_sorted)
        # print("U", U)
        # print("index_to_id", index_to_id)
        # print("id_to_index", id_to_index)

        # Number of nodes remaining in this iteration.
        n_remaining = n_original - iteration

        # Garbage collection.
        if gc and iteration > 0 and iteration % gc == 0:
            nodes_sorted, D_sorted = _rapid_gc(
                nodes_sorted=nodes_sorted,
                D_sorted=D_sorted,
                clustered=clustered,
                obsolete=obsolete,
                n_remaining=n_remaining,
            )

        # before = time.time()

        # Perform one iteration of the neighbour-joining algorithm.
        # u_max, searched, visited = _rapid_iteration(
        u_max = _rapid_iteration(
            iteration=iteration,
            D=D,
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

    #     duration = time.time() - before
    #     timings.append(duration)
    #     searches.append(searched)
    #     visits.append(visited)

    # if diagnostics:
    #     return Z, np.array(timings), np.array(searches), np.array(visits)

    return Z


@numba.njit
def _rapid_gc(
    nodes_sorted: np.ndarray,
    D_sorted: np.ndarray,
    clustered: np.ndarray,
    obsolete: np.ndarray,
    n_remaining: int,
) -> tuple[np.ndarray, np.ndarray]:
    for i in range(nodes_sorted.shape[0]):
        if obsolete[i]:
            continue
        j_new = 0
        for j in range(nodes_sorted.shape[1]):
            id_j = nodes_sorted[i, j]
            if clustered[id_j]:
                continue
            nodes_sorted[i, j_new] = id_j
            D_sorted[i, j_new] = D_sorted[i, j]
            j_new += 1
    nodes_sorted = nodes_sorted[:, :n_remaining]
    D_sorted = D_sorted[:, :n_remaining]
    return nodes_sorted, D_sorted


@numba.njit
def _rapid_iteration(
    iteration: int,
    D: np.ndarray,
    D_sorted: np.ndarray,
    U: np.ndarray,
    nodes_sorted: np.ndarray,
    index_to_id: np.ndarray,
    id_to_index: np.ndarray,
    clustered: np.ndarray,
    obsolete: np.ndarray,
    Z: np.ndarray,
    n_original: int,
    disallow_negative_distances: bool,
    u_max: np.float32,
    # ) -> tuple[np.float32, int, int]:
) -> np.float32:
    # This will be the identifier for the new node to be created in this iteration.
    node = iteration + n_original

    # Number of nodes remaining in this iteration.
    n_remaining = n_original - iteration

    if n_remaining > 2:
        # Search for the closest pair of nodes to join.
        # i_min, j_min, searched, visited = _rapid_search(
        i_min, j_min = _rapid_search(
            D_sorted=D_sorted,
            U=U,
            nodes_sorted=nodes_sorted,
            clustered=clustered,
            obsolete=obsolete,
            index_to_id=index_to_id,
            id_to_index=id_to_index,
            n_remaining=n_remaining,
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
        # searched = 0
        # visited = 0

    # Sanity checks.
    assert child_i >= 0
    assert child_j >= 0
    assert child_i != child_j

    # print("i_min", i_min, "j_min", j_min, "child_i", child_i, "child_j", child_j)

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
        u_max = _rapid_update(
            D=D,
            D_sorted=D_sorted,
            U=U,
            nodes_sorted=nodes_sorted,
            index_to_id=index_to_id,
            id_to_index=id_to_index,
            clustered=clustered,
            obsolete=obsolete,
            node=node,
            child_i=child_i,
            child_j=child_j,
            i_min=i_min,
            j_min=j_min,
            d_ij=d_ij,
        )

    # return u_max, searched, visited
    return u_max


@numba.njit
def _rapid_search(
    D_sorted: np.ndarray,
    U: np.ndarray,
    nodes_sorted: np.ndarray,
    clustered: np.ndarray,
    obsolete: np.ndarray,
    id_to_index: np.ndarray,
    index_to_id: np.ndarray,
    n_remaining: int,
    u_max: np.float32,
    # ) -> tuple[int, int, int, int]:
) -> tuple[int, int]:
    # Initialize working variables.
    q_min = numba.float32(np.inf)
    threshold = numba.float32(np.inf)
    i_min = -1
    j_min = -1
    # searched = 0
    # visited = 0
    coefficient = numba.float32(n_remaining - 2)
    m = nodes_sorted.shape[0]
    n = nodes_sorted.shape[1]
    assert m == D_sorted.shape[0]
    assert n == D_sorted.shape[1]

    # First pass, seed q_min and threshold with first in each row. This is suggested
    # in the original paper, but I'm not sure it actually makes much if any difference
    # in practice.
    for i in range(m):
        if obsolete[i]:
            continue
        u_i = U[i]
        node_j = nodes_sorted[i, 0]
        assert node_j >= 0
        if clustered[node_j]:
            continue
        node_i = index_to_id[i]
        if node_i == node_j:
            continue
        d = D_sorted[i, 0]
        j = id_to_index[node_j]
        u_j = U[j]
        q = coefficient * d - u_i - u_j
        if q < q_min:
            q_min = q
            threshold = q_min + u_max
            i_min = i
            j_min = j

    # Search all values up to threshold.
    for i in range(m):
        # Skip if row is no longer in use.
        if obsolete[i]:
            continue

        # Obtain divergence for node corresponding to this row.
        u_i = U[i]

        # Obtain identifier for node corresponding to the current row.
        node_i = index_to_id[i]

        # Search the row up to threshold.
        for s in range(1, n):
            # visited += 1

            # Obtain node identifier for the current item.
            node_j = nodes_sorted[i, s]

            # Skip if this node is already clustered or self comparison.
            if clustered[node_j] or node_i == node_j:
                continue

            # Break at end of nodes.
            if node_j < 0:
                break

            # Access distance.
            d = D_sorted[i, s]

            # Partially calculate q.
            q_partial = coefficient * d - u_i

            # Limit search. Because the row is sorted, if we are already above this
            # threshold then we know there is no need to search remaining nodes in the
            # row.
            if q_partial > threshold:
                break

            # Fully calculate q.
            j = id_to_index[node_j]
            u_j = U[j]
            q = q_partial - u_j
            # searched += 1

            if q < q_min:
                q_min = q
                threshold = q_min + u_max
                i_min = i
                j_min = j

    # return i_min, j_min, searched, visited
    return i_min, j_min


@numba.njit
def _rapid_update(
    D: np.ndarray,
    D_sorted: np.ndarray,
    U: np.ndarray,
    nodes_sorted: np.ndarray,
    index_to_id: np.ndarray,
    id_to_index: np.ndarray,
    clustered: np.ndarray,
    obsolete: np.ndarray,
    node: int,
    child_i: int,
    child_j: int,
    i_min: int,
    j_min: int,
    d_ij: float,
) -> np.float32:
    # Update data structures. Here we obsolete the row corresponding to the node at
    # j_min, and we reuse the row at i_min for the new node.
    clustered[child_i] = True
    clustered[child_j] = True

    # Assign the new node to row at i_min.
    index_to_id[i_min] = node
    id_to_index[node] = i_min

    # Obsolete the data corresponding to the node at j_min.
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
        D[j_min, k] = np.inf
        D[k, j_min] = np.inf

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
    assert p == distances_new.shape[0]
    nodes_sorted[i_min, :p] = nodes_sorted_new
    nodes_sorted[i_min, p:] = -1
    D_sorted[i_min, :p] = distances_sorted_new
    D_sorted[i_min, p:] = np.inf

    return u_max
