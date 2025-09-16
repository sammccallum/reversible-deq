# Reversible Deep Equilibrium Models

This repository implements Reversible Deep Equilibrium Models - introduced in this paper.

## Library
See [revdeq](https://github.com/sammccallum/revdeq).

## Example
```python
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import reversible_deq as revdeq

class Function(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(2, 1, 10, 2, key=key)

    def __call__(self, z, x):
        x = jnp.concatenate([z, x], axis=0)
        return self.mlp(x)

function = Function(key=jr.PRNGKey(0))
x = jnp.ones(1)
z0 = jnp.zeros(1)
solver = revdeq.Reversible(beta=0.8)
adjoint = revdeq.ReversibleAdjoint()
tol = 1e-3
max_steps = 5

sol = revdeq.solve(function, z0, x, solver, adjoint, tol, max_steps)
z1 = sol.z1
steps = sol.steps
error = sol.error
```

We define a function with signature `(z, args) -> z`, here `args=x`. Then choose a `solver` and `adjoint` with `tol` and `max_steps`. The `sol` object is the solution and contains the fixed point `z1`, number of steps taken `steps` and final residual `error`.

Any gradient taken with respect to `z1` will automatically follow the exact reversible backpropagation algorithm.
