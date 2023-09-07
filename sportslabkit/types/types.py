from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch


""" Types """

Box: TypeAlias = np.ndarray  # Box is of shape (1, 2xdim)
Vector: TypeAlias = np.ndarray | list[Any] | torch.Tensor  # Vector is of shape (1, N)
PathLike: TypeAlias = str | Path  # Path-like objects
NpImage: TypeAlias = np.ndarray  # Numpy/OpenCV image alias
