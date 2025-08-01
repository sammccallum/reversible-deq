import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import reversible_deq as rdeq

from .helpers import tree_allclose


jax.config.update("jax_enable_x64", True)


class F(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self):
        self.mlp = eqx.nn.MLP(2, 1, 10, 2, key=jr.PRNGKey(1))

    def __call__(self, z, x):
        x = jnp.concatenate([z, x], axis=0)
        return self.mlp(x)


class pytree_F(eqx.Module):
    mlp1: eqx.nn.MLP
    mlp2: eqx.nn.MLP

    def __init__(self):
        self.mlp1 = eqx.nn.MLP(2, 1, 3, 1, key=jr.PRNGKey(1))
        self.mlp2 = eqx.nn.MLP(2, 1, 3, 1, key=jr.PRNGKey(2))

    def __call__(self, z, x):
        x1 = jnp.concatenate([z[0], x], axis=0)
        x2 = jnp.concatenate([z[1], x], axis=0)
        return (self.mlp1(x1), self.mlp2(x2))


@eqx.filter_value_and_grad
def grad_loss(vjp_arg, solver, adjoint):
    function, args = vjp_arg
    z0 = jnp.zeros(1)
    sol = rdeq.solve(
        function,
        z0,
        args,
        solver,
        adjoint,
        tol=1e-8,
        max_steps=500,
    )
    return jnp.sum(sol.z1)


@eqx.filter_value_and_grad
def pytree_grad_loss(vjp_arg, z0, solver, adjoint):
    function, args = vjp_arg
    sol = rdeq.solve(
        function,
        z0,
        args,
        solver,
        adjoint,
        tol=1e-8,
        max_steps=500,
    )
    return jnp.sum(sol.z1[0]) + jnp.sum(sol.z1[1])


@pytest.mark.parametrize(
    "solver",
    [
        rdeq.Simple(),
        rdeq.Relaxed(beta=0.8),
    ],
)
def test_implicit_adjoint(solver):
    function = F()
    args = jnp.array([0.1])
    adjoint = rdeq.RecursiveCheckpointAdjoint()
    loss, grads_recursive = grad_loss((function, args), solver, adjoint)

    adjoint = rdeq.ImplicitAdjoint(b_tol=1e-8, b_max_steps=500)
    loss, grads_implicit = grad_loss((function, args), solver, adjoint)

    assert tree_allclose(grads_recursive, grads_implicit, atol=1e-5)


@pytest.mark.parametrize(
    "solver",
    [
        rdeq.Simple(),
        rdeq.Relaxed(beta=0.8),
    ],
)
def test_implicit_adjoint_pytree(solver):
    function = pytree_F()
    args = jnp.array([0.1])
    z0 = (jnp.array([0.2]), jnp.array([0.3]))
    adjoint = rdeq.RecursiveCheckpointAdjoint()
    loss, grads_recursive = pytree_grad_loss((function, args), z0, solver, adjoint)

    adjoint = rdeq.ImplicitAdjoint(b_tol=1e-8, b_max_steps=500)
    loss, grads_implicit = pytree_grad_loss((function, args), z0, solver, adjoint)

    assert tree_allclose(grads_recursive, grads_implicit, atol=1e-5)


@pytest.mark.parametrize("beta", [0.8, 1.2])
def test_reversible_adjoint(beta):
    function = F()
    args = jnp.array([0.1])
    solver = rdeq.Reversible(beta)
    adjoint = rdeq.RecursiveCheckpointAdjoint()
    loss, grads_recursive = grad_loss((function, args), solver, adjoint)

    adjoint = rdeq.ReversibleAdjoint()
    loss, grads_reversible = grad_loss((function, args), solver, adjoint)

    assert tree_allclose(grads_recursive, grads_reversible, atol=1e-5)


@pytest.mark.parametrize("beta", [0.8, 1.2])
def test_reversible_adjoint_pytree(beta):
    function = pytree_F()
    args = jnp.array([0.1])
    z0 = (jnp.array([0.2]), jnp.array([0.3]))
    solver = rdeq.Reversible(beta)
    adjoint = rdeq.RecursiveCheckpointAdjoint()
    loss, grads_recursive = pytree_grad_loss((function, args), z0, solver, adjoint)

    adjoint = rdeq.ReversibleAdjoint()
    loss, grads_reversible = pytree_grad_loss((function, args), z0, solver, adjoint)

    assert tree_allclose(grads_recursive, grads_reversible, atol=1e-5)


def test_phantom_adjoint():
    function = F()
    args = jnp.array([0.1])
    solver = rdeq.Relaxed(beta=0.8)
    adjoint = rdeq.RecursiveCheckpointAdjoint()
    loss, grads_recursive = grad_loss((function, args), solver, adjoint)

    adjoint = rdeq.PhantomAdjoint(beta=0.5, unroll_steps=50)
    loss, grads_phantom = grad_loss((function, args), solver, adjoint)

    assert tree_allclose(grads_recursive, grads_phantom, atol=1e-5)


def test_phantom_adjoint_pytree():
    function = pytree_F()
    args = jnp.array([0.1])
    z0 = (jnp.array([0.0]), jnp.array([0.0]))
    solver = rdeq.Relaxed(beta=0.8)
    adjoint = rdeq.RecursiveCheckpointAdjoint()
    loss, grads_recursive = pytree_grad_loss((function, args), z0, solver, adjoint)
    adjoint = rdeq.PhantomAdjoint(beta=0.5, unroll_steps=50)
    loss, grads_phantom = pytree_grad_loss((function, args), z0, solver, adjoint)

    assert tree_allclose(grads_recursive, grads_phantom, atol=1e-5)
