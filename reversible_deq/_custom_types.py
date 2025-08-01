from typing import Any, Callable

import equinox as eqx
from jaxtyping import ArrayLike, Float, PyTree, Shaped


Z = PyTree[Shaped[ArrayLike, "?*z"]]
Args = PyTree[Any]
Function = Callable[[Z, Args], Z]
FloatScalarLike = Float[ArrayLike, ""]


class Solution(eqx.Module):
    z1: Z
    steps: int
    error: FloatScalarLike
