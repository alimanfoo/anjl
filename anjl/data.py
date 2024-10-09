import numpy as np
import pandas as pd


def example_1():
    # This is example 1 from Amelia Harrison's blog.
    # https://www.tenderisthebyte.com/blog/2022/08/31/neighbor-joining-trees/
    dist = np.array(
        [  # A B C D
            [0, 4, 5, 10],
            [4, 0, 7, 12],
            [5, 7, 0, 9],
            [10, 12, 9, 0],
        ],
        dtype=np.float32,
    )
    leaf_data = pd.DataFrame({"name": ["A", "B", "C", "D"]})
    return dist, leaf_data


def example_2():
    # This is example 2 from Amelia Harrison's blog.
    # https://www.tenderisthebyte.com/blog/2022/08/31/neighbor-joining-trees/
    dist = np.array(
        [  # A B C D
            [0, 2, 2, 2],
            [2, 0, 3, 2],
            [2, 3, 0, 2],
            [2, 2, 2, 0],
        ],
        dtype=np.float32,
    )
    leaf_data = pd.DataFrame({"name": ["A", "B", "C", "D"]})
    return dist, leaf_data


def example_3():
    # This is the extra from Amelia Harrison's blog.
    # https://www.tenderisthebyte.com/blog/2022/08/31/neighbor-joining-trees/
    dist = np.array(
        [  # A B C D E
            [0, 13, 14, 11, 20],
            [13, 0, 7, 12, 13],
            [14, 7, 0, 13, 10],
            [11, 12, 13, 0, 19],
            [20, 13, 10, 19, 0],
        ],
        dtype=np.float32,
    )
    leaf_data = pd.DataFrame({"name": ["A", "B", "C", "D", "E"]})
    return dist, leaf_data


def wikipedia_example():
    # This example comes from the wikipedia page on neighbour-joining.
    # https://en.wikipedia.org/wiki/Neighbor_joining#Example

    dist = np.array(
        [  # a b c d e
            [0, 5, 9, 9, 8],
            [5, 0, 10, 10, 9],
            [9, 10, 0, 8, 7],
            [9, 10, 8, 0, 3],
            [8, 9, 7, 3, 0],
        ],
        dtype=np.float32,
    )
    leaf_data = pd.DataFrame({"name": ["a", "b", "c", "d", "e"]})
    return dist, leaf_data


def mosquitoes():
    import importlib.resources
    from . import resources

    dist_path = importlib.resources.path(resources, "mosquitoes.npy")
    dist = np.load(dist_path)
    leaf_data_path = importlib.resources.path(resources, "mosquitoes.csv")
    leaf_data = pd.read_csv(leaf_data_path)
    return dist, leaf_data
