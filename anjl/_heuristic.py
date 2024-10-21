import numpy as np
from numpy.typing import NDArray

# from numba import njit, uintp, float32, bool_
from numpydoc_decorator import doc
from . import params
from ._util import FLOAT32_INF, UINTP_MAX

# Clausen 2023
# https://doi.org/10.1093/bioinformatics/btac774


@doc(
    summary="""@@TODO.""",
    extended_summary="""
        @@TODO
    """,
)
def heuristic_nj(
    D: params.D,
    disallow_negative_distances: params.disallow_negative_distances = True,
    progress: params.progress = None,
    progress_options: params.progress_options = {},
) -> params.Z:
    """TODO"""

    # Make a copy of distance matrix D because we will overwrite it during the
    # algorithm.
    D_copy: NDArray[np.float32] = np.array(D, copy=True, order="C", dtype=np.float32)
    del D

    # Number of original observations.
    n_original = D_copy.shape[0]

    # Expected number of new (internal) nodes that will be created.
    n_internal = n_original - 1

    # Map row indices to node IDs.
    index_to_id: NDArray[np.uintp] = np.arange(n_original, dtype=np.uintp)

    # Initialise output. This is similar to the output that scipy hierarchical
    # clustering functions return, where each row contains data for one internal node
    # in the tree, except that each row here contains:
    # - left child node ID
    # - right child node ID
    # - distance to left child node
    # - distance to right child node
    # - total number of leaves
    Z: NDArray[np.float32] = np.zeros(shape=(n_internal, 5), dtype=np.float32)

    # Initialize the "divergence" array, containing sum of distances to other nodes.
    S: NDArray[np.float32] = np.sum(D_copy, axis=1)

    # Keep track of which rows correspond to nodes that have been clustered.
    obsolete: NDArray[np.bool_] = np.zeros(shape=n_original, dtype=np.bool_)

    # Initialise the heuristic algorithm.
    J, z = heuristic_init(
        D=D_copy,
        S=S,
        Z=Z,
        obsolete=obsolete,
        index_to_id=index_to_id,
        disallow_negative_distances=disallow_negative_distances,
    )
    # print("init done")
    # print("z", z)
    # print("J", J)

    # Support wrapping the iterator in a progress bar.
    iterator = range(1, n_internal)
    if progress:
        iterator = progress(iterator, **progress_options)

    # Begin iterating.
    for iteration in iterator:
        # print()
        # print("-" * 79)
        # print("iteration", iteration, "z", z)

        # Perform one iteration of the neighbour-joining algorithm.
        z = heuristic_iteration(
            iteration=np.uintp(iteration),
            D=D_copy,
            S=S,
            J=J,
            previous_z=z,
            index_to_id=index_to_id,
            obsolete=obsolete,
            Z=Z,
            n_original=np.uintp(n_original),
            disallow_negative_distances=disallow_negative_distances,
        )

    return Z


# @njit(
#     (
#         float32[:, :],  # D
#         float32[:],  # S
#         float32[:, :],  # Z
#         bool_[:],  # obsolete
#         uintp[:],  # index_to_id
#         bool_,  # disallow_negative_distances
#     ),
#     nogil=NOGIL,
#     fastmath=FASTMATH,
#     error_model=ERROR_MODEL,
#     boundscheck=BOUNDSCHECK,
# )
def heuristic_init(
    D,
    S,
    Z,
    obsolete,
    index_to_id,
    disallow_negative_distances,
):
    # Here we take a first pass through the distance matrix to locate the first pair
    # of nodes to join, and initialise the data structures needed for the heuristic
    # algotithm.
    # print("init")

    # Size of the distance matrix.
    n = np.uintp(D.shape[0])

    # Distance between pair of nodes with global minimum.
    d_xy = FLOAT32_INF

    # Global minimum join criterion.
    q_xy = FLOAT32_INF

    # Indices of the pair of nodes with the global minimum, to be joined.
    x = UINTP_MAX
    y = UINTP_MAX

    # Partially compute outside loop.
    coefficient = np.float32(n - 2)

    # Index of node where minimum join criterion per node, i.e., nearest neighbour
    # within each row.
    J = np.empty(shape=n, dtype=np.uintp)

    # Scan the distance matrix.
    for _i in range(n):
        i = np.uintp(_i)  # row index
        j = UINTP_MAX  # column index of row q minimum
        q_ij = FLOAT32_INF  # row q minimum
        d_ij = FLOAT32_INF  # distance of row q minimum
        s_i = S[i]
        # for _k in range(i):
        for _k in range(n):
            k = np.uintp(_k)
            if i == k:
                continue
            s_k = S[k]
            d = D[i, k]
            q = coefficient * d - s_i - s_k
            if q < q_ij:
                # Found new minimum within this row.
                q_ij = q
                d_ij = d
                j = k
        # Store minimum for this row.
        J[i] = j
        if q_ij < q_xy:
            # Found new global minimum.
            q_xy = q_ij
            d_xy = d_ij
            x = i
            y = j

    # Sanity checks.
    assert x < n
    assert y < n
    assert x != y

    # Stabilise ordering for easier comparisons.
    if x > y:
        x, y = y, x

    # Calculate distances to the new internal node.
    d_xz = 0.5 * (d_xy + (1 / (n - 2)) * (S[x] - S[y]))
    d_yz = 0.5 * (d_xy + (1 / (n - 2)) * (S[y] - S[x]))

    # Handle possibility of negative distances.
    if disallow_negative_distances:
        d_xz = max(np.float32(0), d_xz)
        d_yz = max(np.float32(0), d_yz)

    # Store new node data.
    Z[0, 0] = x
    Z[0, 1] = y
    Z[0, 2] = d_xz
    Z[0, 3] = d_yz
    Z[0, 4] = 2

    # Identifier for the new node.
    parent = n

    # Row index to be used for the new node.
    z = x
    # print("parent", n, "x", x, "y", y, "z", z)

    # Update data structures.
    obsolete[y] = True
    index_to_id[z] = parent

    # Initialize divergence for the new node.
    s_z = np.float32(0)

    # Update distances and divergence.
    for _k in range(D.shape[0]):
        k = np.uintp(_k)

        if k == x or k == y:
            continue

        # Calculate distance from k to the new node.
        d_kx = D[k, x]
        d_ky = D[k, y]
        d_kz = np.float32(0.5) * (d_kx + d_ky - d_xy)
        D[z, k] = d_kz
        D[k, z] = d_kz

        # Subtract out the distances for the nodes that have just been joined and add
        # in distance for the new node.
        s_k = S[k] - d_kx - d_ky + d_kz
        S[k] = s_k

        # Accumulate divergence for the new node.
        s_z += d_kz

    # Assign divergence for the new node.
    S[z] = s_z

    return J, z


def search_row(D, S, J, obsolete, i, coefficient, update=True):
    q_ij = FLOAT32_INF  # row minimum q
    d_ij = FLOAT32_INF  # distance at row minimum q
    j = UINTP_MAX  # column index at row minimum q
    s_i = S[i]  # divergence for node at row i
    # for _k in range(i):
    for _k in range(D.shape[1]):
        k = np.uintp(_k)
        if i == k:
            continue
        if obsolete[k]:
            continue
        s_k = S[k]
        d = D[i, k]
        q = coefficient * d - s_i - s_k
        if q < q_ij:
            # Found new row minimum.
            q_ij = q
            d_ij = d
            j = k
    if update:
        J[i] = j
    return j, q_ij, d_ij


# @njit(
#     (
#         float32[:, :],  # D
#         float32[:],  # S
#         uintp[:],  # J
#         uintp,  # z
#         bool_[:],  # obsolete
#         uintp,  # n_remaining
#     ),
#     nogil=NOGIL,
#     fastmath=FASTMATH,
#     error_model=ERROR_MODEL,
#     boundscheck=BOUNDSCHECK,
# )
def heuristic_search(
    D: NDArray[np.float32],
    S: NDArray[np.float32],
    J,
    z,  # index of new node created in previous iteration
    obsolete: NDArray[np.bool_],
    n_remaining,
):
    # print("begin search")
    # print("z", z)
    # print("J", J)

    # Size of the distance matrix.
    n = np.uintp(D.shape[0])

    # Distance between pair of nodes with global minimum.
    d_xy = FLOAT32_INF

    # Global minimum join criterion.
    q_xy = FLOAT32_INF

    # Indices of the pair of nodes with the global minimum, to be joined.
    x = UINTP_MAX
    y = UINTP_MAX

    # Partially compute outside loop.
    coefficient = np.float32(n_remaining - 2)

    # print(f"First search the rows up to the new node z={z}.")
    # Because these rows do not contain a comparison with the new node at index z,
    # we only need to examine the previous best match in this row.
    for _i in range(1, z):
        i = np.uintp(_i)  # row index
        # print(f"i={i}")
        if obsolete[i]:
            continue

        # Initialise working variables.
        q_ij = FLOAT32_INF  # row minimum q
        d_ij = FLOAT32_INF  # distance at row minimum q

        # Access the best match from the previous iteration.
        j = J[i]  # column index
        # print("Search", i, j)

        if j == UINTP_MAX:
            # print("Edge case, no active nodes to compare at all in this row.")
            pass

        elif obsolete[j] or j == z:
            # print(f"Previous best match j={j} obsolete, rescan row i={i}.")
            j, q_ij, d_ij = search_row(
                D=D, S=S, J=J, obsolete=obsolete, i=i, coefficient=coefficient
            )
            assert j == z, (i, j, z)

        else:
            # print(f"Previous best match still available at row i={i}, col j={j}.")
            s_i = S[i]
            s_j = S[j]
            d_ij = D[i, j]
            q_ij = coefficient * d_ij - s_i - s_j

        if q_ij < q_xy:
            # print(
            #     f"Found new global minimum at i={i}, j={j}, q_ij={q_ij}, d_ij={d_ij}."
            # )
            q_xy = q_ij
            d_xy = d_ij
            x = i
            y = j

    # print(f"Second, fully search the row corresponding to the new node z={z}.")
    i = z
    # print(f"i={i}")
    j, q_ij, d_ij = search_row(
        D=D, S=S, J=J, obsolete=obsolete, i=i, coefficient=coefficient
    )
    if q_ij < q_xy:
        # print(f"Found new global minimum at i={i}, j={j}, q_ij={q_ij}, d_ij={d_ij}.")
        q_xy = q_ij
        d_xy = d_ij
        x = i
        y = j

    # print(f"Third, search all other rows after z={z}.")
    # Calculating q only for the previous best pair of nodes and for the new node at z.
    for _i in range(z + 1, n):
        i = np.uintp(_i)  # row index
        # print(f"i={i}")
        if obsolete[i]:
            continue

        # Initialise working variables.
        q_ij = FLOAT32_INF  # row minimum q
        d_ij = FLOAT32_INF  # distance at row minimum q

        # Access the previous best match.
        j = J[i]
        # print("Search", i, j)

        if j == UINTP_MAX:
            # print("Edge case, no active nodes to compare at all in this row.")
            pass

        elif obsolete[j] or j == z:
            # print(f"Previous best match j={j} obsolete, rescan row i={i}.")
            j, q_ij, d_ij = search_row(
                D=D, S=S, J=J, obsolete=obsolete, i=i, coefficient=coefficient
            )
            assert j == z, (i, j, z)

        else:
            # print(f"Previous best match still available at row i={i}, col j={j}.")
            s_i = S[i]
            s_j = S[j]
            d_ij = D[i, j]
            q_ij = coefficient * d_ij - s_i - s_j

            # print(f"Compare new node at row i={i}, col z={z}")
            d_iz = D[i, z]
            s_z = S[z]
            q_iz = coefficient * d_iz - s_i - s_z
            if q_iz < q_ij:
                j = z
                q_ij = q_iz
                d_ij = d_iz
                J[i] = j

        if q_ij < q_xy:
            # print(
            #     f"Found new global minimum at i={i}, j={j}, q_ij={q_ij}, d_ij={d_ij}."
            # )
            q_xy = q_ij
            d_xy = d_ij
            x = i
            y = j

    return x, y, d_xy


# @njit(
#     (
#         float32[:, :],  # D
#         float32[:],  # S
#         uintp[:],  # index_to_id
#         bool_[:],  # obsolete
#         uintp,  # parent
#         uintp,  # x
#         uintp,  # y
#         float32,  # d_xy
#     ),
#     nogil=NOGIL,
#     fastmath=FASTMATH,
#     error_model=ERROR_MODEL,
#     boundscheck=BOUNDSCHECK,
# )
def heuristic_update(
    D: NDArray[np.float32],
    S: NDArray[np.float32],
    index_to_id: NDArray[np.uintp],
    obsolete: NDArray[np.bool_],
    parent: np.uintp,
    x: np.uintp,
    y: np.uintp,
    d_xy: np.float32,
):
    # Here we obsolete the row and column corresponding to the node at y, and we
    # reuse the row and column at x for the new node.
    obsolete[y] = True

    # Row index to be used for the new node.
    z = x

    # Node identifier.
    index_to_id[z] = parent

    # Initialize divergence for the new node.
    s_z = np.float32(0)

    # Update distances and divergence.
    for _k in range(D.shape[0]):
        k = np.uintp(_k)

        if obsolete[k] or k == x or k == y:
            continue

        # Calculate distance from k to the new node.
        # TODO Only use lower triange of the distance matrix.
        d_kx = D[k, x]
        d_ky = D[k, y]
        d_kz = np.float32(0.5) * (d_kx + d_ky - d_xy)
        D[z, k] = d_kz
        D[k, z] = d_kz

        # Subtract out the distances for the nodes that have just been joined and add
        # in distance for the new node.
        s_k = S[k] - d_kx - d_ky + d_kz
        S[k] = s_k

        # Accumulate divergence for the new node.
        s_z += d_kz

    # Assign divergence for the new node.
    S[z] = s_z

    return z


# @njit(
#     (
#         uintp,  # iteration
#         float32[:, :],  # D
#         float32[:],  # S
#         uintp[:],  # J
#         uintp,  # z
#         uintp[:],  # index_to_id
#         bool_[:],  # obsolete
#         float32[:, :],  # Z
#         uintp,  # n_original
#         bool_,  # disallow_negative_distances
#     ),
#     nogil=NOGIL,
#     fastmath=FASTMATH,
#     error_model=ERROR_MODEL,
#     boundscheck=BOUNDSCHECK,
# )
def heuristic_iteration(
    iteration: np.uintp,
    D: NDArray[np.float32],
    S: NDArray[np.float32],
    J,
    previous_z,
    index_to_id: NDArray[np.uintp],
    obsolete: NDArray[np.bool_],
    Z: NDArray[np.float32],
    n_original: np.uintp,
    disallow_negative_distances: bool,
):
    # This will be the identifier for the new node to be created in this iteration.
    parent = iteration + n_original

    # Number of nodes remaining in this iteration.
    n_remaining = n_original - iteration

    if n_remaining > 2:
        # Search for the closest pair of nodes to join.
        x, y, d_xy = heuristic_search(
            D=D, S=S, J=J, z=previous_z, obsolete=obsolete, n_remaining=n_remaining
        )
        assert x < D.shape[0], x
        assert y < D.shape[0], y
        assert not np.isinf(d_xy), d_xy

        # Calculate distances to the new internal node.
        d_xz = 0.5 * (d_xy + (1 / (n_remaining - 2)) * (S[x] - S[y]))
        d_yz = 0.5 * (d_xy + (1 / (n_remaining - 2)) * (S[y] - S[x]))

    else:
        # Termination. Join the two remaining nodes, placing the final node at the
        # midpoint.
        _x, _y = np.nonzero(~obsolete)[0]
        x = np.uintp(_x)
        y = np.uintp(_y)
        d_xy = D[x, y]
        d_xz = d_xy / 2
        d_yz = d_xy / 2

    # Handle possibility of negative distances.
    if disallow_negative_distances:
        d_xz = max(np.float32(0), d_xz)
        d_yz = max(np.float32(0), d_yz)

    # Get IDs for the nodes to be joined.
    child_x = index_to_id[x]
    child_y = index_to_id[y]

    # Sanity checks.
    assert x < D.shape[0]
    assert y < D.shape[0]
    assert x != y
    assert child_x != child_y

    # Stabilise ordering for easier comparisons.
    if child_x > child_y:
        child_x, child_y = child_y, child_x
        x, y = y, x
        d_xz, d_yz = d_yz, d_xz

    # Get number of leaves.
    if child_x < n_original:
        leaves_i = np.float32(1)
    else:
        leaves_i = Z[child_x - n_original, 4]
    if child_y < n_original:
        leaves_j = np.float32(1)
    else:
        leaves_j = Z[child_y - n_original, 4]

    # Store new node data.
    Z[iteration, 0] = child_x
    Z[iteration, 1] = child_y
    Z[iteration, 2] = d_xz
    Z[iteration, 3] = d_yz
    Z[iteration, 4] = leaves_i + leaves_j

    if n_remaining > 2:
        # Update data structures.
        new_z = heuristic_update(
            D=D,
            S=S,
            index_to_id=index_to_id,
            obsolete=obsolete,
            parent=parent,
            x=x,
            y=y,
            d_xy=d_xy,
        )

    else:
        new_z = UINTP_MAX

    return new_z
