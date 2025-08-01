import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree

from .._base import AbstractSolver
from .._custom_types import FloatScalarLike, Function, Z


_SolverState = Z


class Relaxed(AbstractSolver[_SolverState]):
    beta: float = 0.8

    def init(self, function: Function, z0: Z, args: PyTree) -> _SolverState:
        return function(z0, args)

    def step(
        self, function: Function, z0: Z, args: PyTree, solver_state: _SolverState
    ) -> tuple[Z, _SolverState, FloatScalarLike]:
        f0 = solver_state
        z1 = jtu.tree_map(lambda z, f: (1 - self.beta) * z + self.beta * f, z0, f0)
        f1 = function(z1, args)
        error = jtu.tree_reduce(
            jnp.maximum,
            jtu.tree_map(
                lambda z, f: jnp.linalg.norm(z - f) / (1e-5 + jnp.linalg.norm(f)),
                z1,
                f1,
            ),
        )
        return z1, f1, error
