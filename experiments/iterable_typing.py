# %%
# from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def is_array_like(obj):
    return isinstance(obj, ArrayLike)


print("Should be True:")
print(is_array_like([1, 2, 3]))
print(is_array_like((1, 2, 3)))
print(is_array_like({1, 2, 3}))
print(is_array_like(np.array([1, 2, 3])))
print(is_array_like(pd.Series([1, 2, 3])))
print(is_array_like(pd.Index([1, 2, 3])))
print()
print("Should be False:")
print(is_array_like("123"))
print(is_array_like(123))
print(is_array_like(None))
print(is_array_like(True))

# %%
