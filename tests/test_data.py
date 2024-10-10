import anjl
import numpy as np
import pandas as pd


def test_mosquitoes():
    D, leaf_data = anjl.data.mosquitoes()
    assert isinstance(D, np.ndarray)
    assert D.ndim == 2
    assert isinstance(leaf_data, pd.DataFrame)
