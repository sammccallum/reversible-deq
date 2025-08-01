import equinox as eqx
import equinox.internal as eqxi
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω

from ._base import AbstractAdjoint, AbstractSolver
from ._custom_types import Args, Function, Solution, Z
from ._solver import Relaxed, Reversible


def _forward_loop(vjp_arg, z0, solver, tol, max_steps, loop_kind):
    function, args = vjp_arg

    def cond_fun(state):
        _, _, step, error = state
        max_check = step < max_steps
        error_check = error > tol
        return jnp.logical_and(max_check, error_check)

    def body_fun(state):
        z0, solver_state, step, error = state
        step += 1
        z1, solver_state, error = solver.step(function, z0, args, solver_state)
        return z1, solver_state, step, error.astype(float)

    solver_state = solver.init(function, z0, args)
    step = 0
    error = 2 * tol  # start error above tol
    state = (z0, solver_state, step, error)

    state = eqxi.while_loop(
        cond_fun, body_fun, state, max_steps=max_steps, kind=loop_kind
    )

    z1, solver_state, steps, error = state
    sol = Solution(z1, steps, error)
    return sol, state


class RecursiveCheckpointAdjoint(AbstractAdjoint):
    def loop(
        self,
        function: Function,
        z0: Z,
        args: Args,
        solver: AbstractSolver,
        tol: float,
        max_steps: int,
    ) -> Solution:
        sol, _ = _forward_loop(
            (function, args), z0, solver, tol, max_steps, loop_kind="checkpointed"
        )
        return sol


@eqx.filter_custom_vjp
def _implicit_loop(vjp_arg, z0, solver, tol, max_steps, b_tol, b_max_steps):
    del b_tol, b_max_steps
    sol, _ = _forward_loop(vjp_arg, z0, solver, tol, max_steps, loop_kind="lax")
    return sol


@_implicit_loop.def_fwd
def _implicit_loop_fwd(
    perturbed, vjp_arg, z0, solver, tol, max_steps, b_tol, b_max_steps
):
    del perturbed, b_tol, b_max_steps
    sol, final_state = _forward_loop(
        vjp_arg, z0, solver, tol, max_steps, loop_kind="lax"
    )
    return sol, final_state


@_implicit_loop.def_bwd
def _implicit_loop_bwd(
    residuals,
    grad_sol,
    perturbed,
    vjp_arg,
    z0,
    solver,
    tol,
    max_steps,
    b_tol,
    b_max_steps,
):
    del perturbed, z0
    function, args = vjp_arg
    z1, _, _, _ = residuals
    grad_z1 = grad_sol.z1
    f_call = lambda f, z, args: f(z, args)
    _, grad_z1_fun = eqx.filter_vjp(f_call, function, z1, args)
    grad_fun = lambda g, args: (ω(grad_z1_fun(g)[1]) + ω(grad_z1)).ω
    grad_sol = _implicit_loop(
        (grad_fun, None), grad_z1, solver, b_tol, b_max_steps, tol, max_steps
    )
    g1 = grad_sol.z1
    grad_function, _, grad_args = grad_z1_fun(g1)
    return grad_function, grad_args


class ImplicitAdjoint(AbstractAdjoint):
    b_tol: float = 1e-8
    b_max_steps: int = 100

    def loop(
        self,
        function: Function,
        z0: Z,
        args: Args,
        solver: AbstractSolver,
        tol: float,
        max_steps: int,
    ) -> Solution:
        return _implicit_loop(
            (function, args), z0, solver, tol, max_steps, self.b_tol, self.b_max_steps
        )


@eqx.filter_custom_vjp
def _reversible_loop(vjp_arg, z0, solver, tol, max_steps):
    sol, _ = _forward_loop(vjp_arg, z0, solver, tol, max_steps, loop_kind="lax")
    return sol


@_reversible_loop.def_fwd
def _reversible_loop_fwd(perturbed, vjp_arg, z0, solver, tol, max_steps):
    del perturbed
    sol, state = _forward_loop(vjp_arg, z0, solver, tol, max_steps, loop_kind="lax")
    return sol, state


@_reversible_loop.def_bwd
def _reversible_loop_bwd(
    residuals,
    grad_sol,
    perturbed,
    vjp_arg,
    z0,
    solver,
    tol,
    max_steps,
):
    del perturbed, z0, tol, max_steps
    function, args = vjp_arg
    z1, (y1, f1), steps, _ = residuals
    beta = solver.beta
    f_call = lambda f, z, args: f(z, args)

    def cond_fun(state):
        step, *rest = state
        return step > 0

    def body_fun(state):
        step, z1, y1, grad_z1, grad_y1, grad_function1, grad_args1 = state

        # Backward step
        z0 = jtu.tree_map(
            lambda z, f: (z - beta * f) / (1 - beta), z1, f_call(function, y1, args)
        )
        y0 = jtu.tree_map(
            lambda y, f: (y - beta * f) / (1 - beta), y1, f_call(function, z0, args)
        )

        # Gradients
        _, grad_fun_y1 = eqx.filter_vjp(f_call, function, y1, args)
        _, grad_fun_z0 = eqx.filter_vjp(f_call, function, z0, args)
        dgrad_function_y1, dgrad_y1, dgrad_args_y1 = grad_fun_y1(grad_z1)

        grad_y1 = (ω(grad_y1) + beta * ω(dgrad_y1)).ω
        grad_y0 = ((1 - beta) * ω(grad_y1)).ω

        dgrad_function_z0, dgrad_z0, dgrad_args_z0 = grad_fun_z0(grad_y1)
        grad_z0 = ((1 - beta) * ω(grad_z1) + beta * ω(dgrad_z0)).ω
        grad_function0 = eqx.apply_updates(
            grad_function1,
            jtu.tree_map(
                lambda tree1, tree2: beta * (tree1 + tree2),
                dgrad_function_y1,
                dgrad_function_z0,
            ),
        )
        grad_args0 = eqx.apply_updates(
            grad_args1,
            jtu.tree_map(
                lambda tree1, tree2: beta * (tree1 + tree2),
                dgrad_args_y1,
                dgrad_args_z0,
            ),
        )
        step -= 1
        return step, z0, y0, grad_z0, grad_y0, grad_function0, grad_args0

    grad_z1 = grad_sol.z1
    grad_y1 = jtu.tree_map(jnp.zeros_like, y1)
    diff_function = eqx.filter(function, eqx.is_inexact_array)
    grad_function1 = jtu.tree_map(jnp.zeros_like, diff_function)
    diff_args = eqx.filter(args, eqx.is_inexact_array)
    grad_args1 = jtu.tree_map(jnp.zeros_like, diff_args)

    state = (steps, z1, y1, grad_z1, grad_y1, grad_function1, grad_args1)
    state = eqxi.while_loop(cond_fun, body_fun, state, kind="lax")
    step, z0, y0, grad_z0, grad_y0, grad_function, grad_args = state

    return grad_function, grad_args


class ReversibleAdjoint(AbstractAdjoint):
    def loop(
        self,
        function: Function,
        z0: Z,
        args: Args,
        solver: Reversible,
        tol: float,
        max_steps: int,
    ) -> Solution:
        if not isinstance(solver, Reversible):
            raise ValueError(
                "`ReversibleAdjoint` is only compatible with `solver=Reversible()`."
            )
        return _reversible_loop((function, args), z0, solver, tol, max_steps)


@eqx.filter_custom_vjp
def _phantom_loop(vjp_arg, z0, solver, tol, max_steps, beta, unroll_steps):
    del beta, unroll_steps
    sol, _ = _forward_loop(vjp_arg, z0, solver, tol, max_steps, loop_kind="lax")
    return sol


@_phantom_loop.def_fwd
def _phantom_loop_fwd(
    perturbed,
    vjp_arg,
    z0,
    solver,
    tol,
    max_steps,
    beta,
    unroll_steps,
):
    del perturbed, beta, unroll_steps
    sol, final_state = _forward_loop(
        vjp_arg, z0, solver, tol, max_steps, loop_kind="lax"
    )
    return sol, final_state


@_phantom_loop.def_bwd
def _phantom_loop_bwd(
    residuals,
    grad_sol,
    perturbed,
    vjp_arg,
    z0,
    solver,
    tol,
    max_steps,
    beta,
    unroll_steps,
):
    del perturbed, z0, tol, max_steps
    function, args = vjp_arg
    z1, _, _, _ = residuals
    grad_z1 = grad_sol.z1
    solver = Relaxed(beta)

    def _phantom_solve(function, args):
        sol, _ = _forward_loop(
            (function, args),
            z1,
            solver,
            tol=1e-15,
            max_steps=unroll_steps,
            loop_kind="checkpointed",
        )
        return sol.z1

    _, grad_solve = eqx.filter_vjp(_phantom_solve, function, args)
    grad_function, grad_args = grad_solve(grad_z1)

    return grad_function, grad_args


class PhantomAdjoint(AbstractAdjoint):
    beta: float = 0.5
    unroll_steps: int = 10

    def loop(
        self,
        function: Function,
        z0: Z,
        args: Args,
        solver: AbstractSolver,
        tol: float,
        max_steps: int,
    ) -> Solution:
        return _phantom_loop(
            (function, args),
            z0,
            solver,
            tol,
            max_steps,
            self.beta,
            self.unroll_steps,
        )
