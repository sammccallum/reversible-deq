from functools import partial

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

from .._base import AbstractSolver
from .._custom_types import Args, Function, Z


_SolverState = tuple


def _shift_circular_buffer(buffer, new_val):
    buffer = buffer.at[:-1].set(buffer[1:])
    buffer = buffer.at[-1].set(new_val)
    return buffer


@eqx.filter_jit
def _update(zks, fks, beta, m):
    gks = fks - zks
    G = gks.reshape((m + 1, -1))
    H = G @ G.T + 1e-6 * jnp.identity(m + 1)
    b = jnp.ones(m + 1)
    alpha = jnp.linalg.lstsq(H, b)[0]
    alpha = alpha / jnp.sum(alpha)
    z1 = (1 - beta) * jnp.einsum("i,i...->...", alpha, zks) + beta * jnp.einsum(
        "i,i...->...", alpha, fks
    )
    return z1


class Anderson(AbstractSolver[_SolverState]):
    beta: float = 0.8
    m: int = 1

    def init(self, function: Function, z0: Z, args: Args) -> _SolverState:
        # Initialise buffers
        f0 = function(z0, args)
        zks = jtu.tree_map(lambda z: jnp.array([z for m in range(self.m + 1)]), z0)
        fks = jtu.tree_map(lambda f: jnp.array([f for m in range(self.m + 1)]), f0)

        return zks, fks

    def step(
        self, function: Function, z0: Z, args: Args, solver_state: _SolverState
    ) -> tuple[Z, _SolverState, float]:
        zks, fks = solver_state

        # Update state
        f0 = function(z0, args)
        zks = jtu.tree_map(_shift_circular_buffer, zks, z0)
        fks = jtu.tree_map(_shift_circular_buffer, fks, f0)

        # Solve residual least-squares
        z1 = jtu.tree_map(partial(_update, beta=self.beta, m=self.m), zks, fks)
        solver_state = (zks, fks)

        gk = jtu.tree_map(
            lambda _zks, _fks: jnp.linalg.norm(_fks[-1] - _zks[-1])
            / (1e-5 + jnp.linalg.norm(_fks[-1])),
            zks,
            fks,
        )
        error = jtu.tree_reduce(jnp.maximum, gk)

        return z1, solver_state, error  # type: ignore
