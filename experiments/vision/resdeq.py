import os


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import torch
import torchvision
import wandb
from _models import ImplicitResNet


def loss_fn(model, state, xs, ys, key, gamma=1.0):
    keys = jr.split(key, xs.shape[0])
    ys_pred, state, regs = eqx.filter_vmap(
        model, axis_name="batch", in_axes=(0, None, 0), out_axes=(0, None, 0)
    )(xs, state, keys)
    y_pred = jnp.take_along_axis(ys_pred, jnp.expand_dims(ys, 1), axis=1)
    return jnp.mean(-y_pred[:, 0] + gamma * regs), state


@eqx.filter_jit
def make_step(model, xs, ys, key, optim, opt_state, state):
    key, subkey = jr.split(key)
    (loss, state), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, state, xs, ys, key
    )
    updates, opt_state = optim.update(
        grads, opt_state, eqx.filter(model, eqx.is_inexact_array), value=loss
    )
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, subkey


@eqx.filter_jit
def compute_accuracy(model, state, xs, ys):
    key = jr.PRNGKey(0)
    keys = jr.split(key, xs.shape[0])
    inference_model = eqx.nn.inference_mode(model)
    ys_pred, _, _ = eqx.filter_vmap(
        inference_model, in_axes=(0, None, 0), out_axes=(0, None, 0)
    )(xs, state, keys)
    ys_pred = jnp.argmax(ys_pred, axis=1)
    return jnp.mean(ys_pred == ys)


def get_model(config):
    model_config = config["model"]
    key = jr.PRNGKey(model_config["seed"])
    data_shape = model_config["data_shape"]
    channels = model_config["channels"]
    width_expansions = model_config["width_expansions"]
    steps = model_config["steps"]
    beta = model_config["beta"]
    tol = model_config["tol"]

    model, state = eqx.nn.make_with_state(ImplicitResNet)(
        data_shape, channels, width_expansions, steps, beta, tol, key
    )
    return model, state


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


class Logger:
    t0: float

    def __init__(self):
        self.t0 = time.time()

    def _time(self):
        current_time = time.time()
        dt = current_time - self.t0
        self.t0 = current_time
        return dt

    def _accuracy(self, model, state, dataloader, num_batches):
        total = 0
        acc = 0
        for i, (xs, ys) in zip(range(num_batches), dataloader):
            xs = jax.tree.map(lambda tensor: tensor.numpy(), xs)
            ys = jax.tree.map(lambda tensor: tensor.numpy(), ys)
            acc += compute_accuracy(model, state, xs, ys)
            total += 1
        acc = 100 * acc / total
        return acc

    def log(self, run, epoch, model, state, trainloader, testloader, scale):
        dt = self._time()
        acc = self._accuracy(model, state, testloader, num_batches=len(testloader))
        train_acc = self._accuracy(model, state, trainloader, num_batches=10)
        run.log(
            {
                "epoch": epoch,
                "dt": dt,
                "train accuracy": train_acc,
                "test accuracy": acc,
                "lr scale": scale,
            }
        )
        print(
            f"Epoch: {epoch}, train accuracy: {train_acc:.3f}, test accuracy: {acc:.3f}, time: {dt:.3f}, lr scale: {scale:.3e}"
        )


def train(config):
    run = wandb.init(project="Reversible DEQ", config=config)

    model, state = get_model(config)
    params = config["hyperparameters"]

    epochs = params["epochs"]
    lr = params["lr"]
    batch_size = params["batch_size"]

    trainloader, testloader = get_dataloaders(batch_size)

    optim = optax.chain(
        optax.adamw(lr),
        optax.contrib.reduce_on_plateau(factor=0.5, patience=10, accumulation_size=200),
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    key = jr.PRNGKey(5678)
    step = 0
    logger = Logger()
    for epoch in range(1, epochs + 1):
        for xs, ys in trainloader:
            xs = jax.tree.map(lambda tensor: tensor.numpy(), xs)
            ys = jax.tree.map(lambda tensor: tensor.numpy(), ys)
            model, state, opt_state, key = make_step(
                model, xs, ys, key, optim, opt_state, state
            )
            scale = opt_state[-1].scale
            step += 1
        logger.log(run, epoch, model, state, trainloader, testloader, scale)
        if scale <= 1e-3:
            break

    run.finish()


if __name__ == "__main__":
    model_config = {
        "seed": 5680,
        "data_shape": [3, 32, 32],
        "channels": [128, 256, 256, 128],
        "width_expansions": [1, 3, 3, 1],
        "steps": [4, 4, 4, 4],
        "beta": 0.8,
        "tol": 1e-6,
    }
    hyperparameter_config = {"epochs": 100, "lr": 3e-4, "batch_size": 64}
    config = {"model": model_config, "hyperparameters": hyperparameter_config}
    train(config)
