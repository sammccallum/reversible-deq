import jax


jax.config.update("jax_enable_x64", True)

import math

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import reversible_deq as rdeq


W_DTYPE = jnp.float32
Z_DTYPE = jnp.float64


class ImplicitResidualLayer(eqx.Module):
    conv1: eqx.nn.Conv2d | eqx.nn.WeightNorm
    conv2: eqx.nn.Conv2d | eqx.nn.WeightNorm
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm
    norm3: eqx.nn.GroupNorm

    def __init__(self, channels, width_expansion, weight_norm=False, *, key):
        keys = jr.split(key, 2)
        conv1 = eqx.nn.Conv2d(
            channels,
            width_expansion * channels,
            kernel_size=3,
            padding=1,
            use_bias=False,
            key=keys[0],
            dtype=W_DTYPE,
        )

        conv2 = eqx.nn.Conv2d(
            width_expansion * channels,
            channels,
            kernel_size=3,
            padding=1,
            use_bias=False,
            key=keys[1],
            dtype=W_DTYPE,
        )

        if weight_norm:
            self.conv1 = eqx.nn.WeightNorm(conv1)
            self.conv2 = eqx.nn.WeightNorm(conv2)

        else:
            self.conv1 = conv1
            self.conv2 = conv2

        groups = channels // 16
        self.norm1 = eqx.nn.GroupNorm(
            width_expansion * groups, width_expansion * channels, dtype=W_DTYPE
        )
        self.norm2 = eqx.nn.GroupNorm(groups, channels, dtype=W_DTYPE)
        self.norm3 = eqx.nn.GroupNorm(groups, channels, dtype=W_DTYPE)

    def __call__(self, z, x):
        y = jax.nn.relu(self.norm1(self.conv1(z.astype(W_DTYPE))))
        return self.norm3(jax.nn.relu(z + self.norm2(x + self.conv2(y)))).astype(
            Z_DTYPE
        )


class DownsampleLayer(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    norm1: eqx.nn.BatchNorm
    norm2: eqx.nn.BatchNorm
    residual_conv: eqx.nn.Conv2d
    residual_norm: eqx.nn.BatchNorm

    def __init__(self, in_channels, out_channels, key):
        keys = jr.split(key)
        self.conv1 = eqx.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            use_bias=False,
            key=keys[0],
            dtype=W_DTYPE,
        )
        self.conv2 = eqx.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            key=keys[1],
            dtype=W_DTYPE,
        )
        self.norm1 = eqx.nn.BatchNorm(
            out_channels, axis_name="batch", mode="batch", dtype=W_DTYPE
        )
        self.norm2 = eqx.nn.BatchNorm(
            out_channels, axis_name="batch", mode="batch", dtype=W_DTYPE
        )

        self.residual_conv = eqx.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=2,
            use_bias=False,
            key=keys[2],
            dtype=W_DTYPE,
        )
        self.residual_norm = eqx.nn.BatchNorm(
            out_channels, axis_name="batch", mode="batch", dtype=W_DTYPE
        )

    def __call__(self, x, state):
        h, state = self.norm1(self.conv1(x), state)
        h = jax.nn.relu(h)
        h, state = self.norm2(self.conv2(h), state)
        x = self.residual_conv(x)
        x, state = self.residual_norm(x, state)
        h += x
        h = jax.nn.relu(h)
        return h, state


class DownsampleLayerPreAct(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    norm1: eqx.nn.BatchNorm
    norm2: eqx.nn.BatchNorm
    residual_conv: eqx.nn.Conv2d

    def __init__(self, in_channels, out_channels, key):
        keys = jr.split(key)
        self.conv1 = eqx.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            use_bias=False,
            key=keys[0],
            dtype=W_DTYPE,
        )
        self.conv2 = eqx.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            key=keys[1],
            dtype=W_DTYPE,
        )
        self.norm1 = eqx.nn.BatchNorm(
            in_channels, axis_name="batch", mode="batch", dtype=W_DTYPE
        )
        self.norm2 = eqx.nn.BatchNorm(
            out_channels, axis_name="batch", mode="batch", dtype=W_DTYPE
        )

        self.residual_conv = eqx.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=2,
            use_bias=False,
            key=keys[2],
            dtype=W_DTYPE,
        )

    def __call__(self, x, state):
        h, state = self.norm1(x, state)
        h = jax.nn.relu(h)
        h = self.conv1(h)

        h, state = self.norm2(h, state)
        h = jax.nn.relu(h)
        h = self.conv2(h)

        x = self.residual_conv(x)
        h += x

        return h, state


class DEQ(eqx.Module):
    function: ImplicitResidualLayer
    solver: rdeq.AbstractSolver
    adjoint: rdeq.AbstractAdjoint
    tol: float
    max_steps: int

    def __init__(self, function, solver, adjoint, tol, max_steps):
        self.function = function
        self.solver = solver
        self.adjoint = adjoint
        self.tol = tol
        self.max_steps = max_steps

    def __call__(self, x):
        z0 = jnp.zeros_like(x, dtype=Z_DTYPE)
        sol = rdeq.solve(
            self.function,
            z0,
            x,
            self.solver,
            self.adjoint,
            self.tol,
            self.max_steps,
        )
        z_star = sol.z1

        return z_star.astype(W_DTYPE)


class DEQ_Regularisation(eqx.Module):
    function: ImplicitResidualLayer
    solver: rdeq.AbstractSolver
    adjoint: rdeq.AbstractAdjoint
    tol: float
    max_steps: int
    reg: bool

    def __init__(self, function, solver, adjoint, tol, max_steps, reg=False):
        self.function = function
        self.solver = solver
        self.adjoint = adjoint
        self.tol = tol
        self.max_steps = max_steps
        self.reg = reg

    def _compute_regularisation(self, x_in, z_star, key):
        f_z = lambda z: self.function(z, x_in)
        _, vjp_fun = eqx.filter_vjp(f_z, z_star)
        d = math.prod(z_star.shape)
        eps = jr.normal(key, z_star.shape)
        reg = jnp.sum(vjp_fun(eps)[0] ** 2) / d
        return reg

    def __call__(self, x, key):
        z0 = jnp.zeros_like(x, dtype=Z_DTYPE)
        sol = rdeq.solve(
            self.function,
            z0,
            x,
            self.solver,
            self.adjoint,
            self.tol,
            self.max_steps,
        )
        z_star = sol.z1.astype(W_DTYPE)

        if self.reg:
            reg = self._compute_regularisation(x, z_star, key)
        else:
            reg = 0

        return z_star, reg


class ImplicitResNet(eqx.Module):
    encoder_conv: eqx.nn.Conv2d
    encoder_norm: eqx.nn.BatchNorm
    deq_layers: list[DEQ_Regularisation]
    down_layers: list[DownsampleLayerPreAct]
    pool: eqx.nn.AvgPool2d
    final_linear: eqx.nn.Linear

    def __init__(
        self,
        data_shape,
        channels,
        width_expansions,
        steps,
        beta,
        tol,
        key,
    ):
        assert len(channels) == len(steps)
        num_layers = len(channels)
        encoder_key, deq_key, down_key, linear_key = jr.split(key, 4)
        deq_keys = jr.split(deq_key, num_layers)
        down_keys = jr.split(down_key, num_layers - 1)

        self.encoder_conv = eqx.nn.Conv2d(
            data_shape[0],
            channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            key=encoder_key,
            dtype=W_DTYPE,
        )
        self.encoder_norm = eqx.nn.BatchNorm(
            channels[0], axis_name="batch", mode="batch", dtype=W_DTYPE
        )

        solver = rdeq.Reversible(beta)
        adjoint = rdeq.ReversibleAdjoint()

        self.deq_layers = []
        self.down_layers = []
        for i in range(num_layers):
            res = ImplicitResidualLayer(
                channels[i], width_expansions[i], weight_norm=False, key=deq_keys[i]
            )
            max_steps = steps[i]
            deq = DEQ_Regularisation(res, solver, adjoint, tol, max_steps, reg=False)
            self.deq_layers.append(deq)

            if i < num_layers - 1:
                down = DownsampleLayerPreAct(channels[i], channels[i + 1], down_keys[i])
                self.down_layers.append(down)

        final_width = data_shape[1] // 2 ** (num_layers - 1)
        self.pool = eqx.nn.AvgPool2d(final_width, final_width)
        self.final_linear = eqx.nn.Linear(
            channels[-1], 10, key=linear_key, dtype=W_DTYPE
        )

    def __call__(self, x, state, key):
        key1, key2 = jr.split(key)
        x = self.encoder_conv(x)
        x, state = self.encoder_norm(x, state)
        x = jax.nn.relu(x)

        reg = 0
        for deq_layer, down_layer in zip(self.deq_layers[:-1], self.down_layers):
            dx, dreg = deq_layer(x, key1)
            reg += dreg
            x = x + dx
            x, state = down_layer(x, state)

        dx, dreg = self.deq_layers[-1](x, key2)
        reg += dreg
        x = x + dx
        x = self.pool(x)
        x = self.final_linear(x.ravel())
        x = jax.nn.log_softmax(x)

        return x, state, reg


class SingleDEQ(eqx.Module):
    encoder_conv: eqx.nn.Conv2d
    encoder_norm: eqx.nn.GroupNorm
    deq_model: DEQ
    decoder_pool: eqx.nn.AdaptiveAvgPool2d
    decoder_layer: eqx.nn.Linear
    reg: bool
    n_hutch: int

    def __init__(
        self, data_channels, in_channels, deq_model, key, reg=False, n_hutch=1
    ):
        keys = jr.split(key, 2)
        self.reg = reg
        self.n_hutch = n_hutch

        # Encoder
        self.encoder_conv = eqx.nn.Conv2d(
            data_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            key=keys[0],
            dtype=W_DTYPE,
        )
        groups = in_channels // 16
        self.encoder_norm = eqx.nn.GroupNorm(groups, in_channels, dtype=W_DTYPE)

        # DEQ
        self.deq_model = deq_model

        # Decoder
        final_dim = 4
        self.decoder_pool = eqx.nn.AdaptiveAvgPool2d(final_dim)
        self.decoder_layer = eqx.nn.Linear(
            in_channels * final_dim * final_dim, 10, key=keys[1], dtype=W_DTYPE
        )

    def _compute_regularisation(self, x_in, z_star, key):
        keys = jr.split(key, self.n_hutch)
        f_z = lambda z: self.deq_model.function(z, x_in)
        _, vjp_fun = eqx.filter_vjp(f_z, z_star)
        d = math.prod(z_star.shape)
        reg = 0
        for n in range(self.n_hutch):
            eps = jr.normal(keys[n], z_star.shape)
            reg += jnp.sum(vjp_fun(eps)[0] ** 2) / d

        return reg / self.n_hutch

    def __call__(self, x, key):
        x_in = self.encoder_norm(self.encoder_conv(x))
        z_star = self.deq_model(x_in)
        x_out = self.decoder_pool(z_star).ravel()
        logits = self.decoder_layer(x_out)
        y = jax.nn.log_softmax(logits)

        if self.reg:
            reg = self._compute_regularisation(x_in, z_star, key)
        else:
            reg = 0

        return y, reg
