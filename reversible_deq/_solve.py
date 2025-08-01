from jaxtyping import PyTree

from ._adjoint import RecursiveCheckpointAdjoint
from ._base import AbstractAdjoint, AbstractSolver
from ._custom_types import Function, Solution, Z


def solve(
    function: Function,
    z0: Z,
    args: PyTree,
    solver: AbstractSolver,
    adjoint: AbstractAdjoint = RecursiveCheckpointAdjoint(),
    tol: float = 1e-3,
    max_steps: int = 50,
) -> Solution:
    return adjoint.loop(function, z0, args, solver, tol, max_steps)
