"""Init file.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def round_to_integer(x: ArrayLike) -> np.ndarray[np.int32]:
    return np.round(x).astype(np.int32)
