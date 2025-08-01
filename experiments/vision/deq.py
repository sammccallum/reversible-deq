import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
import torch
import torchvision
import wandb
from _models import (
    DEQ,
    ImplicitResidualLayer,
    SingleDEQ,
)

import reversible_deq as rdeq


@eqx.filter_value_and_grad
def grad_loss(model, xs, ys, key, gamma=1.0):
    batch_size = xs.shape[0]
    keys = jr.split(key, batch_size)
    ys_pred, regs = eqx.filter_vmap(model, in_axes=(0, 0))(xs, keys)
    y_pred = jnp.take_along_axis(ys_pred, jnp.expand_dims(ys, 1), axis=1)
    return jnp.mean(-y_pred[:, 0] + gamma * regs)


@eqx.filter_jit
def make_step(model, xs, ys, optim, opt_state, key):
    key, return_key = jr.split(key)
    loss, grads = grad_loss(model, xs, ys, key)
    updates, opt_state = optim.update(
        grads, opt_state, eqx.filter(model, eqx.is_inexact_array), value=loss
    )
    model = eqx.apply_updates(model, updates)
    return model, loss, opt_state, return_key


@eqx.filter_jit
def compute_accuracy(model, xs, ys, key):
    batch_size = xs.shape[0]
    keys = jr.split(key, batch_size)
    ys_pred, _ = eqx.filter_vmap(model, in_axes=(0, 0))(xs, keys)
    ys_pred = jnp.argmax(ys_pred, axis=1)
    return jnp.mean(ys_pred == ys)


def get_dataloaders(batch_size):
    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ),
        ]
    )
    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        "../../datasets",
        train=True,
        download=True,
        transform=transform_train,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        "../../datasets",
        train=False,
        download=True,
        transform=transform_test,
    )
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )
    return trainloader, testloader


def swap_adjoint(model, adjoint):
    is_adjoint = lambda x: isinstance(x, rdeq.AbstractAdjoint)
    where = lambda m: [
        x for x in jtu.tree_leaves(m, is_leaf=is_adjoint) if is_adjoint(x)
    ]
    replace_fn = lambda _: adjoint
    model = eqx.tree_at(
        where,
        model,
        replace_fn=replace_fn,
    )
    return model


class Logger:
    t0: float
    epochs: list
    accs: list
    times: list

    def __init__(self):
        self.t0 = time.time()
        self.epochs = []
        self.accs = []
        self.times = []

    def _epoch(self, epoch):
        self.epochs.append(epoch)

    def _time(self):
        current_time = time.time()
        self.times.append(current_time - self.t0)
        self.t0 = current_time

    def _accuracy(self, model, dataloader, key):
        total = 0
        acc = 0
        for xs, ys in dataloader:
            xs = jax.tree.map(lambda tensor: tensor.numpy(), xs)
            ys = jax.tree.map(lambda tensor: tensor.numpy(), ys)
            acc += compute_accuracy(model, xs, ys, key)
            total += 1
        acc = 100 * acc / total
        self.accs.append(acc)

    def _train_accuracy(self, model, dataloader, num_batches, key):
        total = 0
        acc = 0
        for i, (xs, ys) in zip(range(num_batches), dataloader):
            xs = jax.tree.map(lambda tensor: tensor.numpy(), xs)
            ys = jax.tree.map(lambda tensor: tensor.numpy(), ys)
            acc += compute_accuracy(model, xs, ys, key)
            total += 1
        acc = 100 * acc / total
        return acc

    def log(self, run, epoch, model, trainloader, testloader, scale, key):
        key1, key2 = jr.split(key)
        self._epoch(epoch)
        self._time()
        self._accuracy(model, testloader, key1)
        train_acc = self._train_accuracy(model, trainloader, num_batches=10, key=key2)
        run.log(
            {
                "epoch": epoch,
                "time": self.times[-1],
                "train accuracy": train_acc,
                "test accuracy": self.accs[-1],
                "lr scale": scale,
            }
        )
        print(
            f"Epoch: {epoch}, train accuracy: {train_acc:.3f}, test accuracy: {self.accs[-1]:.5f}, time: {self.times[-1]:.3f}, lr scale: {scale:.3f}"
        )


def train(config, key):
    training_config = config["training"]
    model_config = config["model"]
    solver_config = config["solver"]

    run = wandb.init(project="Reversible DEQ", config=config)

    data_channels = model_config["data_channels"]
    channels = model_config["channels"]
    width_expansion = model_config["width"]

    trainloader, testloader = get_dataloaders(training_config["batch_size"])

    f_key, model_key, training_key = jr.split(key, 3)
    solver = rdeq.Reversible(beta=solver_config["beta"])
    adjoint = rdeq.ReversibleAdjoint()
    function = ImplicitResidualLayer(
        channels, width_expansion, weight_norm=model_config["weight_norm"], key=f_key
    )
    deq = DEQ(
        function,
        solver=solver,
        adjoint=adjoint,
        tol=solver_config["tol"],
        max_steps=solver_config["max_steps"],
    )
    reg = model_config["reg"]

    model = SingleDEQ(data_channels, channels, deq, key=model_key, reg=reg)

    lr = training_config["lr"]
    optim = optax.chain(
        optax.adamw(lr),
        optax.contrib.reduce_on_plateau(factor=0.5, patience=10, accumulation_size=200),
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    epochs = training_config["epochs"]
    step = 0
    logger = Logger()
    for epoch in range(1, epochs + 1):
        for xs, ys in trainloader:
            xs = jax.tree.map(lambda tensor: tensor.numpy(), xs)
            ys = jax.tree.map(lambda tensor: tensor.numpy(), ys)
            model, loss, opt_state, training_key = make_step(
                model, xs, ys, optim, opt_state, training_key
            )
            scale = opt_state[-1].scale
            step += 1
        logger.log(run, epoch, model, trainloader, testloader, scale, training_key)
        if scale <= 1e-3:
            break

    run.finish()


if __name__ == "__main__":
    training_config = {"epochs": 100, "batch_size": 64, "lr": 1e-3}
    model_config = {
        "data_channels": 3,
        "channels": 32,
        "width": 1,
        "reg": False,
        "weight_norm": False,
    }
    solver_config = {
        "beta": 0.8,
        "tol": 1e-6,
        "max_steps": 2,
    }
    config = {
        "training": training_config,
        "model": model_config,
        "solver": solver_config,
    }
    train(config, key=jr.PRNGKey(5678))
