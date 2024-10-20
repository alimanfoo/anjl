import numpy as np
from numpy.typing import NDArray

# from numba import njit, uintp, float32, bool_
from numpydoc_decorator import doc
from . import params
from ._util import FLOAT32_INF, UINTP_MAX


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
    U: NDArray[np.float32] = np.sum(D_copy, axis=1)

    # Keep track of which rows correspond to nodes that have been clustered.
    obsolete: NDArray[np.bool_] = np.zeros(shape=n_original, dtype=np.bool_)

    # Initialise the heuristic algorithm.
    J, z = heuristic_init(
        D=D_copy,
        U=U,
        Z=Z,
        obsolete=obsolete,
        index_to_id=index_to_id,
        disallow_negative_distances=disallow_negative_distances,
    )
    print("init done")
    print("z", z)
    print("J", J)

    # Support wrapping the iterator in a progress bar.
    iterator = range(1, n_internal)
    if progress:
        iterator = progress(iterator, **progress_options)

    # Begin iterating.
    for iteration in iterator:
        print("iteration", iteration, "z", z)

        # Perform one iteration of the neighbour-joining algorithm.
        z = heuristic_iteration(
            iteration=iteration,
            D=D_copy,
            U=U,
            J=J,
            previous_z=z,
            index_to_id=index_to_id,
            obsolete=obsolete,
            Z=Z,
            n_original=n_original,
            disallow_negative_distances=disallow_negative_distances,
        )

    return Z


# @njit(
#     (
#         float32[:, :],  # D
#         float32[:],  # U
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
    U,
    Z,
    obsolete,
    index_to_id,
    disallow_negative_distances,
):
    # Here we take a first pass through the distance matrix to locate the first pair
    # of nodes to join, and initialise the data structures needed for the heuristic
    # algotithm.
    print("init")

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

    # Scan the lower triangle of the distance matrix.
    for _i in range(n):
        i = np.uintp(_i)
        row_q_min = FLOAT32_INF
        row_j_min = UINTP_MAX
        u_i = U[i]
        for _j in range(i):
            j = np.uintp(_j)
            u_j = U[j]
            d = D[i, j]
            q = coefficient * d - u_i - u_j
            if q < row_q_min:
                # Found new minimum within this row.
                row_q_min = q
                row_j_min = j
            if q < q_xy:
                # Found new global minimum.
                q_xy = q
                d_xy = d
                x = i
                y = j
        # Store minimum for this row.
        J[i] = row_j_min

    # Sanity checks.
    assert x < n
    assert y < n
    assert x != y

    # Calculate distances to the new internal node.
    d_xz = 0.5 * (d_xy + (1 / (n - 2)) * (U[x] - U[y]))
    d_yz = 0.5 * (d_xy + (1 / (n - 2)) * (U[y] - U[x]))

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
    print("parent", n, "z", z, "x", x, "y", y)

    # Update data structures.
    obsolete[y] = True
    index_to_id[z] = parent

    # Initialize divergence for the new node.
    u_z = np.float32(0)

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
        u_k = U[k] - d_kx - d_ky + d_kz
        U[k] = u_k

        # Accumulate divergence for the new node.
        u_z += d_kz

    # Assign divergence for the new node.
    U[z] = u_z

    return J, z


# @njit(
#     (
#         float32[:, :],  # D
#         float32[:],  # U
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
    U: NDArray[np.float32],
    J,
    z,  # index of new node created in previous iteration
    obsolete: NDArray[np.bool_],
    n_remaining,
):
    print("begin search")
    print("z", z)
    print("J", J)

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

    print("First search the rows up to the new node z.")
    for _i in range(1, z):
        i = np.uintp(_i)  # row index
        if obsolete[i]:
            continue
        u_i = U[i]
        # Assume that the best match in the previous iteration still holds.
        j = J[i]  # column index
        print("search", i, j)

        if obsolete[j]:
            print("best match obsolete, rescan row i=", i)
            q_ij = FLOAT32_INF
            d_ij = FLOAT32_INF
            j = UINTP_MAX
            for _k in range(i):
                k = np.uintp(_k)
                if obsolete[k]:
                    continue
                u_k = U[k]
                d = D[i, k]
                q = coefficient * d - u_i - u_k
                if q < q_ij:
                    # Found new row minimum.
                    q_ij = q
                    d_ij = d
                    j = k
            J[i] = j

        else:
            print("best match still available")
            u_j = U[j]
            d_ij = D[i, j]
            q_ij = coefficient * d_ij - u_i - u_j

        if q_ij < q_xy:
            # Found new global minimum.
            q_xy = q_ij
            d_xy = d_ij
            x = i
            y = j

    print("Second, fully search the row corresponding to the new node z.")
    i = z
    j = UINTP_MAX
    u_i = U[i]
    d_ij = FLOAT32_INF
    q_ij = FLOAT32_INF
    for _k in range(i):
        k = np.uintp(_k)  # column index
        if obsolete[k]:
            continue
        print("search", i, k)
        u_k = U[k]
        d = D[i, k]
        q = coefficient * d - u_i - u_k
        if q < q_ij:
            # Found new minimum within this row.
            q_ij = q
            d_ij = d
            j = k
    if q_ij < q_xy:
        # Found new global minimum.
        q_xy = q_ij
        d_xy = d_ij
        x = z
        y = k
    # Store best match for this row.
    J[i] = j

    print("Third, search all other rows after z.")
    # Calculating q only for the previous best pair of nodes and for the new node at z.
    for _i in range(z + 1, n):
        i = np.uintp(_i)  # row index
        if obsolete[i]:
            continue
        u_i = U[i]

        # Try the previous best match.
        j = J[i]
        print("search", i, j)
        if obsolete[j]:
            print("best match obsolete, rescan row i=", i)
            q_ij = FLOAT32_INF
            d_ij = FLOAT32_INF
            j = UINTP_MAX
            for _k in range(i):
                k = np.uintp(_k)
                if obsolete[k]:
                    continue
                u_k = U[k]
                d = D[i, k]
                q = coefficient * d - u_i - u_k
                if q < q_ij:
                    # Found new row minimum.
                    q_ij = q
                    d_ij = d
                    j = k
            J[i] = j

        else:
            print("best match still available", i, j)
            u_j = U[j]
            d_ij = D[i, j]
            q_ij = coefficient * d_ij - u_i - u_j

            print("compare new node", i, z)
            d_iz = D[i, z]
            u_z = U[z]
            q_iz = coefficient * d_iz - u_i - u_z
            if q_iz < q_ij:
                j = z
                q_ij = q_iz
                d_ij = d_iz
                J[i] = z

        if q_ij < q_xy:
            # Found new global minimum.
            q_xy = q_ij
            d_xy = d_ij
            x = i
            y = j

    return x, y, d_xy


# @njit(
#     (
#         float32[:, :],  # D
#         float32[:],  # U
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
    U: NDArray[np.float32],
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
    u_z = np.float32(0)

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
        u_k = U[k] - d_kx - d_ky + d_kz
        U[k] = u_k

        # Accumulate divergence for the new node.
        u_z += d_kz

    # Assign divergence for the new node.
    U[z] = u_z

    return z


# @njit(
#     (
#         uintp,  # iteration
#         float32[:, :],  # D
#         float32[:],  # U
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
    U: NDArray[np.float32],
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
            D=D, U=U, J=J, z=previous_z, obsolete=obsolete, n_remaining=n_remaining
        )

        # Calculate distances to the new internal node.
        d_xz = 0.5 * (d_xy + (1 / (n_remaining - 2)) * (U[x] - U[y]))
        d_yz = 0.5 * (d_xy + (1 / (n_remaining - 2)) * (U[y] - U[x]))

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
            U=U,
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
