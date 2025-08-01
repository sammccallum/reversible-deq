from abc import abstractmethod
from typing import Generic, TypeVar

import equinox as eqx

from ._custom_types import Args, FloatScalarLike, Function, Solution, Z


_SolverState = TypeVar("_SolverState")


class AbstractSolver(eqx.Module, Generic[_SolverState]):
    @abstractmethod
    def init(self, function: Function, z0: Z, args: Args) -> _SolverState:
        pass

    @abstractmethod
    def step(
        self, function: Function, z0: Z, args: Args, solver_state: _SolverState
    ) -> tuple[Z, _SolverState, FloatScalarLike]:
        pass


class AbstractAdjoint(eqx.Module):
    @abstractmethod
    def loop(
        self,
        function: Function,
        z0: Z,
        args: Args,
        solver: AbstractSolver,
        tol: float,
        max_steps: int,
    ) -> Solution:
        pass
