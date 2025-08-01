import os


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

import jax


jax.config.update("jax_enable_x64", True)

import itertools
import math
import time

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import wandb
from jaxtyping import Array
from transformers import AutoTokenizer

import reversible_deq as rdeq
from datasets import load_dataset


W_DTYPE = jnp.float32
Z_DTYPE = jnp.float64


class SelfAttention(eqx.Module):
    W_q: eqx.nn.Linear
    W_k: eqx.nn.Linear
    W_v: eqx.nn.Linear
    attention: eqx.nn.MultiheadAttention
    mask: Array

    def __init__(self, seq_length, embedding_size, n_heads, key):
        keys = jr.split(key, 4)
        self.W_q = eqx.nn.Linear(
            embedding_size, embedding_size, key=keys[0], dtype=W_DTYPE
        )
        self.W_k = eqx.nn.Linear(
            embedding_size, embedding_size, key=keys[1], dtype=W_DTYPE
        )
        self.W_v = eqx.nn.Linear(
            embedding_size, embedding_size, key=keys[2], dtype=W_DTYPE
        )
        self.attention = eqx.nn.MultiheadAttention(
            n_heads, embedding_size, key=keys[3], dtype=W_DTYPE
        )
        self.mask = jnp.tril(jnp.ones((seq_length, seq_length), dtype=W_DTYPE))

    def __call__(self, x):
        Q = eqx.filter_vmap(self.W_q)(x)
        K = eqx.filter_vmap(self.W_k)(x)
        V = eqx.filter_vmap(self.W_v)(x)
        output = self.attention(Q, K, V, mask=self.mask)
        return output


class Block(eqx.Module):
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    attention: SelfAttention
    mlp: eqx.nn.MLP

    def __init__(self, seq_length, embedding_size, mlp_hidden_size, n_heads, key):
        keys = jr.split(key, 2)
        self.norm1 = eqx.nn.LayerNorm(embedding_size, dtype=W_DTYPE)
        self.norm2 = eqx.nn.LayerNorm(embedding_size, dtype=W_DTYPE)
        self.attention = SelfAttention(seq_length, embedding_size, n_heads, keys[0])
        self.mlp = eqx.nn.MLP(
            embedding_size,
            embedding_size,
            mlp_hidden_size,
            depth=2,
            activation=jax.nn.gelu,
            key=keys[1],
            dtype=W_DTYPE,
        )

    def __call__(self, z, x):
        z = z.astype(W_DTYPE)
        z = z + self.attention(eqx.filter_vmap(self.norm1)(z + x))
        z = z + eqx.filter_vmap(self.mlp)(eqx.filter_vmap(self.norm2)(z))
        return z.astype(Z_DTYPE)


class DEQ(eqx.Module):
    function: Block
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
        return sol.z1.astype(W_DTYPE)


class Model(eqx.Module):
    seq_length: int
    pos_emb: eqx.nn.Embedding
    tok_emb: eqx.nn.Embedding
    deq: DEQ
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear

    def __init__(
        self,
        vocab_size,
        seq_length,
        embedding_size,
        n_heads,
        solver,
        adjoint,
        tol,
        max_steps,
        key,
    ):
        keys = jr.split(key, 4)
        self.seq_length = seq_length
        self.pos_emb = eqx.nn.Embedding(
            seq_length, embedding_size, key=keys[0], dtype=W_DTYPE
        )
        self.tok_emb = eqx.nn.Embedding(
            vocab_size, embedding_size, key=keys[1], dtype=W_DTYPE
        )

        block = Block(seq_length, embedding_size, 4 * embedding_size, n_heads, keys[2])
        self.deq = DEQ(block, solver, adjoint, tol, max_steps)

        self.norm = eqx.nn.LayerNorm(embedding_size, dtype=W_DTYPE)
        self.head = eqx.nn.Linear(
            embedding_size, vocab_size, use_bias=False, key=keys[3], dtype=W_DTYPE
        )

    def __call__(self, x):
        t = jnp.arange(self.seq_length)
        pos = eqx.filter_vmap(self.pos_emb)(t)
        tok = eqx.filter_vmap(self.tok_emb)(x)
        x = pos + tok
        z = self.deq(x)
        logits = eqx.filter_vmap(self.head)(z)
        log_p = jax.nn.log_softmax(logits)
        return log_p


def dataloader(tokens, batch_size, seq_length):
    dataset_size = len(tokens)
    split_size = batch_size * seq_length
    while True:
        start = 0
        end = split_size
        while end < dataset_size:
            batch = tokens[start : end + 1]
            x = batch[:-1].reshape((batch_size, seq_length))
            y = batch[1:].reshape((batch_size, seq_length))
            yield x, y
            start = end
            end = start + split_size


@eqx.filter_jit
def loss_fn(model, xs, ys):
    log_p = eqx.filter_vmap(model)(xs)
    log_p = jnp.take_along_axis(log_p, jnp.expand_dims(ys, 2), axis=2)
    return -jnp.mean(log_p)


def grad_loss(model, xs, ys):
    return eqx.filter_value_and_grad(loss_fn)(model, xs, ys)


@eqx.filter_jit
def make_step(model, xs, ys, optim, opt_state):
    loss, grads = grad_loss(model, xs, ys)
    updates, opt_state = optim.update(
        grads, opt_state, eqx.filter(model, eqx.is_inexact_array), value=loss
    )
    model = eqx.apply_updates(model, updates)
    return model, loss, opt_state


def get_tokens(dataset):
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="../../datasets")
    tokens = dataset.map(
        lambda batch: tokenizer("\n\n".join(batch["text"]), return_tensors="np"),
        batched=True,
        batch_size=1024,
        remove_columns=["text"],
        num_proc=8,  # pyright: ignore
    )

    tokens_iter = itertools.chain.from_iterable(tokens["train"]["input_ids"])  # pyright: ignore
    train_tokens = np.fromiter(tokens_iter, dtype=np.uint16)

    tokens_iter = itertools.chain.from_iterable(tokens["validation"]["input_ids"])  # pyright: ignore
    val_tokens = np.fromiter(tokens_iter, dtype=np.uint16)

    tokens_iter = itertools.chain.from_iterable(tokens["test"]["input_ids"])  # pyright: ignore
    test_tokens = np.fromiter(tokens_iter, dtype=np.uint16)

    return train_tokens, val_tokens, test_tokens


if __name__ == "__main__":
    config = {
        "seed": 5678,
        "vocab_size": 50304,
        "seq_length": 448,
        "embedding_size": 768,
        "n_heads": 12,
        "beta": 0.5,
        "tol": 1e-3,
        "max_steps": 4,
        "lr": 3e-4,
        "batch_size": 32,
    }
    run = wandb.init(project="Reversible DEQ", config=config)

    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-103-v1", cache_dir="../../datasets"
    )

    train_tokens, val_tokens, test_tokens = get_tokens(dataset)
    train_size, val_size, test_size = (
        len(train_tokens),
        len(val_tokens),
        len(test_tokens),
    )

    vocab_size = config["vocab_size"]
    seq_length = config["seq_length"]
    embedding_size = config["embedding_size"]
    n_heads = config["n_heads"]
    solver = rdeq.Reversible(config["beta"])
    adjoint = rdeq.ReversibleAdjoint()
    tol = config["tol"]
    max_steps = config["max_steps"]
    key = jr.PRNGKey(config["seed"])
    model = Model(
        vocab_size,
        seq_length,
        embedding_size,
        n_heads,
        solver,
        adjoint,
        tol,
        max_steps,
        key,
    )

    optim = optax.chain(
        optax.adamw(config["lr"]),
        optax.contrib.reduce_on_plateau(factor=0.5, patience=10, accumulation_size=200),
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    batch_size = config["batch_size"]
    tokens_per_batch = batch_size * seq_length
    train_steps_per_epoch = math.ceil(train_size / tokens_per_batch)
    val_steps_per_epoch = math.ceil(val_size / tokens_per_batch)
    test_steps_per_epoch = math.ceil(test_size / tokens_per_batch)

    train_loader = dataloader(train_tokens, batch_size, seq_length)
    val_loader = dataloader(val_tokens, batch_size, seq_length)
    test_loader = dataloader(test_tokens, batch_size, seq_length)

    epochs = 10
    print_every = 100
    tokens_processed = print_every * tokens_per_batch
    tic = time.time()
    for epoch in range(epochs):
        for step, (x, y) in zip(range(train_steps_per_epoch), train_loader):
            model, loss, opt_state = make_step(model, x, y, optim, opt_state)
            if step % print_every == 0:
                toc = time.time()
                dt = toc - tic
                x, y = next(iter(val_loader))
                val_loss = loss_fn(model, x, y)
                perplexity = jnp.exp(loss)
                val_perplexity = jnp.exp(val_loss)
                scale = opt_state[-1].scale

                run.log(
                    {
                        "step": step,
                        "dt": dt,
                        "loss": loss,
                        "perplexity": perplexity,
                        "val loss": val_loss,
                        "val perplexity": val_perplexity,
                        "lr scale": scale,
                    }
                )
                print(
                    f"step: {step} | loss: {loss:.5f} | perplexity: {perplexity:.5f} | val loss: {val_loss:.5f} | val perplexity: {val_perplexity:.5f} | tokens/s: {(tokens_processed / dt):.0f}"
                )
                tic = time.time()

        test_loss = 0
        count = 0
        for _, (x, y) in zip(range(test_steps_per_epoch), test_loader):
            test_loss += loss_fn(model, x, y)
            count += 1

        test_loss = test_loss / count
        test_perplexity = jnp.exp(test_loss)

        print(
            f"Epoch: {epoch} | test loss: {test_loss:.5f} | test perplexity {test_perplexity:.5f}"
        )
        run.log(
            {
                "epoch": epoch,
                "test loss": test_loss,
                "test perplexity:": test_perplexity,
            }
        )

    run.finish()
