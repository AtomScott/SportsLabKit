from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

""" types """

# Box is of shape (1,2xdim), e.g. for dim=2 [xmin, ymin, width, height] format is accepted
Box = np.ndarray

# Vector is of shape (1, N)
Vector = np.ndarray

_pathlike = Union[str, Path]

# numpy/opencv image alias
NpImage = np.ndarray
