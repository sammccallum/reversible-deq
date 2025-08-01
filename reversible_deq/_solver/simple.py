import jax.numpy as jnp
import jax.tree_util as jtu

from .._base import AbstractSolver
from .._custom_types import Args, FloatScalarLike, Function, Z


_SolverState = Z


class Simple(AbstractSolver[_SolverState]):
    def init(self, function: Function, z0: Z, args: Args) -> _SolverState:
        return function(z0, args)

    def step(
        self, function: Function, z0: Z, args: Args, solver_state: _SolverState
    ) -> tuple[Z, _SolverState, FloatScalarLike]:
        f0 = solver_state
        z1 = f0
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
