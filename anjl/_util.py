import os
import numpy as np
from numpy.typing import NDArray
from numba import njit, float32, uintp, intp
from . import params


# Common configuration for numba jitted functions.
NOGIL = True
FASTMATH = False  # setting True actually seems to slow things down
ERROR_MODEL = "numpy"
# Detect whether we are running via pytest, if so run with boundscheck enabled to catch
# any out of bounds errors.
# https://docs.pytest.org/en/stable/example/simple.html#detect-if-running-from-within-a-pytest-run
if os.environ.get("PYTEST_VERSION") is not None:
    BOUNDSCHECK = True
else:
    BOUNDSCHECK = False


# Convenience constants.
UINTP_MAX = np.uintp(np.iinfo(np.uintp).max)
FLOAT32_INF = np.float32(np.inf)


def to_string(Z: params.Z) -> str:
    # Total number of internal nodes.
    n_internal = Z.shape[0]

    # Total number of leaf nodes.
    n_original = n_internal + 1

    # Set up the first node to visit, which will be the root node.
    root = n_original + n_internal - 1

    # Initialise working variables.
    text = ""
    stack = [(root, 0, "")]

    # Start processing nodes.
    while stack:
        # Access the next node to process.
        node, dist, indent = stack.pop()
        if node < n_original:
            # Leaf node.
            text += f"{indent}Leaf(id={node}, dist={dist})\n"
        else:
            # Internal node.
            z = node - n_original
            left = int(Z[z, 0])
            right = int(Z[z, 1])
            ldist = Z[z, 2]
            rdist = Z[z, 3]
            count = int(Z[z, 4])
            text += f"{indent}Node(id={node}, dist={dist}, count={count})\n"

            # Put them on the stack in this order so the left node comes out first.
            stack.append((right, rdist, indent + "    "))
            stack.append((left, ldist, indent + "    "))

    return text.strip()


def map_internal_to_leaves(Z: params.Z) -> list[list[int]]:
    # For each internal node, build a list of all the descendant leaf ids.
    index: list[list[int]] = []

    # Total number of internal nodes.
    n_internal = Z.shape[0]

    # Total number of leaf nodes.
    n_original = n_internal + 1

    # Iterate over internal nodes.
    for z in range(n_internal):
        # Create a list to store the leaves for this node.
        leaves = []

        # Access the direct children.
        left = int(Z[z, 0])
        right = int(Z[z, 1])

        # Add to the leaves.
        if left < n_original:
            leaves.append(left)
        else:
            leaves.extend(index[left - n_original])
        if right < n_original:
            leaves.append(right)
        else:
            leaves.extend(index[right - n_original])

        # Store the leaves in the index.
        index.append(leaves)

    return index


@njit(
    float32[::1](
        float32[:, :],
    ),
    nogil=NOGIL,
    fastmath=FASTMATH,
    error_model=ERROR_MODEL,
    boundscheck=BOUNDSCHECK,
)
def square_to_condensed(D):
    """Convert a square distance matrix into a condensed distance matrix in upper
    triangle format as returned by scipy's pdist."""

    # Calculate number of pairs.
    n_original = D.shape[0]
    n_pairs = n_original * (n_original - 1) // 2

    # Allocate condensed distance matrix.
    dist = np.empty(n_pairs, dtype=np.float32)

    # Copy data from square to condensed.
    c = np.uintp(0)
    for _i in range(n_original):
        i = np.uintp(_i)
        for _j in range(i + 1, n_original):
            j = np.uintp(_j)
            dist[c] = D[i, j]
            c += np.uintp(1)

    return dist


def ensure_condensed_distance(
    D: params.D,
    copy: params.copy,
) -> tuple[NDArray[np.float32], int]:
    if D.ndim == 1:
        # Calculate number of original observations.
        n_pairs = D.shape[0]
        n_original = int((1 + np.sqrt(1 + 8 * n_pairs)) // 2)

        # Sanity check.
        if (n_original * (n_original - 1) // 2) != n_pairs:
            raise ValueError("D is not a valid condensed distance matrix.")

        # Assume that D is already a condensed distance matrix in scipy (upper triangle)
        # layout.
        distance = np.array(D, copy=copy, dtype=np.float32)

    elif D.ndim == 2:
        # Check square.
        if D.shape[0] != D.shape[1]:
            raise ValueError("D is not a valid square distance matrix.")

        # Create and fill condensed distance matrix.
        n_original = D.shape[0]
        distance = square_to_condensed(D)

    return distance, n_original


@njit(
    float32[::1](
        float32[::1],
        uintp,
    ),
    nogil=NOGIL,
    fastmath=FASTMATH,
    error_model=ERROR_MODEL,
    boundscheck=BOUNDSCHECK,
)
def setup_divergence(distance, n_original):
    # R is the row sum of distances, i.e., for each node, the sum of distances to all
    # other nodes.
    R = np.zeros(n_original, dtype=np.float32)
    c = np.uintp(0)
    for _i in range(n_original):
        i = np.uintp(_i)
        for _j in range(i + 1, n_original):
            j = np.uintp(_j)
            d = distance[c]
            R[i] += d
            R[j] += d
            c += np.uintp(1)
    return R


@njit(
    uintp(
        intp,  # i
        intp,  # j
        intp,  # n
    ),
    nogil=NOGIL,
    fastmath=FASTMATH,
    error_model=ERROR_MODEL,
    boundscheck=BOUNDSCHECK,
    inline="always",
)
def condensed_index(i: np.intp, j: np.intp, n: np.intp):
    """Convert distance matrix coordinates from square form (i, j) to condensed form."""
    # N.B., need to calculate as signed integers to avoid overflow errors.
    if i > j:
        i, j = j, i  # upper triangle only
    c = (n * i) - (i * (i + 1) // 2) - 1 - i + j

    # Return an unsigned integer for faster indexing.
    return np.uintp(c)


@njit(
    uintp(
        intp,  # i
        intp,  # n
    ),
    nogil=NOGIL,
    fastmath=FASTMATH,
    error_model=ERROR_MODEL,
    boundscheck=BOUNDSCHECK,
    inline="always",
)
def condensed_offset(i: np.intp, n: np.intp):
    c = (n * i) - (i * (i + 1) // 2) - 1 - i

    # Return an unsigned integer for faster indexing.
    return np.uintp(c)
