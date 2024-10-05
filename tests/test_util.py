from textwrap import dedent
import numpy as np
import anjl


def test_to_string():
    # This example comes from Amelia Harrison's blog.
    # https://www.tenderisthebyte.com/blog/2022/08/31/neighbor-joining-trees/

    D = np.array(
        [  # A B C D
            [0, 4, 5, 10],
            [4, 0, 7, 12],
            [5, 7, 0, 9],
            [10, 12, 9, 0],
        ],
        dtype=np.float32,
    )
    Z = anjl.canonical_nj(D)

    # String value.
    expected_str = dedent("""
        Node(id=6, dist=0, count=4)
            Leaf(id=3, dist=3.5)
            Node(id=5, dist=3.5, count=3)
                Leaf(id=2, dist=2.0)
                Node(id=4, dist=2.0, count=2)
                    Leaf(id=0, dist=1.0)
                    Leaf(id=1, dist=3.0)
    """).strip()
    assert anjl.to_string(Z) == expected_str


# TODO test leaf_index()
