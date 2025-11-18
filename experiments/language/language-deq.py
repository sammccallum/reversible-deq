import os


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

import jax


jax.config.update("jax_enable_x64", True)

import time
from collections import Counter

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import Array

import reversible_deq as rdeq
import wandb
from datasets import load_dataset


W_DTYPE = jnp.float32
Z_DTYPE = jnp.float64


class SelfAttention(eqx.Module):
    W_q: eqx.nn.Linear
    W_k: eqx.nn.Linear
    W_v: eqx.nn.Linear
    attention: eqx.nn.MultiheadAttention
    mask: Array

    def __init__(self, seq_length, embedding_size, n_heads, dropout_p, key):
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
            n_heads, embedding_size, dropout_p=dropout_p, key=keys[3], dtype=W_DTYPE
        )
        self.mask = jnp.tril(jnp.ones((seq_length, seq_length), dtype=W_DTYPE))

    def __call__(self, x, *, key):
        Q = eqx.filter_vmap(self.W_q)(x)
        K = eqx.filter_vmap(self.W_k)(x)
        V = eqx.filter_vmap(self.W_v)(x)
        output = self.attention(Q, K, V, mask=self.mask, key=key)
        return output


class Block(eqx.Module):
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    attention: SelfAttention
    mlp: eqx.nn.MLP
    mlp_dropout: eqx.nn.Dropout

    def __init__(
        self, seq_length, embedding_size, mlp_hidden_size, n_heads, dropout_p, key
    ):
        keys = jr.split(key, 4)
        self.norm1 = eqx.nn.LayerNorm(embedding_size, dtype=W_DTYPE)
        self.norm2 = eqx.nn.LayerNorm(embedding_size, dtype=W_DTYPE)
        self.attention = SelfAttention(
            seq_length, embedding_size, n_heads, dropout_p, keys[0]
        )
        self.mlp = eqx.nn.MLP(
            embedding_size,
            embedding_size,
            mlp_hidden_size,
            depth=2,
            activation=jax.nn.gelu,
            key=keys[1],
            dtype=W_DTYPE,
        )
        self.mlp_dropout = eqx.nn.Dropout(dropout_p)

    def __call__(self, z, x_and_key):
        z = z.astype(W_DTYPE)
        x, key = x_and_key
        key1, key2 = jr.split(key)
        z = z + self.attention(eqx.filter_vmap(self.norm1)(z + x), key=key1)
        z_mlp = eqx.filter_vmap(self.mlp)(eqx.filter_vmap(self.norm2)(z))
        z_mlp = self.mlp_dropout(z_mlp, key=key2)
        z = z + z_mlp
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

    def __call__(self, x, *, key):
        z0 = jnp.zeros_like(x, dtype=Z_DTYPE)
        sol = rdeq.solve(
            self.function,
            z0,
            (x, key),
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
    output_scale: Array

    def __init__(
        self,
        vocab_size,
        seq_length,
        embedding_size,
        n_heads,
        dropout_p,
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

        block = Block(
            seq_length, embedding_size, 4 * embedding_size, n_heads, dropout_p, keys[2]
        )
        self.deq = DEQ(block, solver, adjoint, tol, max_steps)

        self.norm = eqx.nn.LayerNorm(embedding_size, dtype=W_DTYPE)
        self.output_scale = 1.0 / jnp.sqrt(embedding_size)

    def __call__(self, x, key):
        t = jnp.arange(self.seq_length)
        pos = eqx.filter_vmap(self.pos_emb)(t)
        tok = eqx.filter_vmap(self.tok_emb)(x)
        x = pos + tok
        z = self.deq(x, key=key)
        z = eqx.filter_vmap(self.norm)(z)
        logits = (z @ self.tok_emb.weight.T) * self.output_scale
        log_p = jax.nn.log_softmax(logits, axis=-1)
        return log_p


def dataloader(tokens, batch_size, seq_length, key):
    dataset_size = len(tokens)
    split_size = batch_size * seq_length
    num_batches = (dataset_size - 1) // split_size

    def generator():
        nonlocal key
        while True:
            key, subkey = jr.split(key)
            indices = jr.permutation(subkey, num_batches)

            for idx in indices:
                start = int(idx * split_size)
                end = start + split_size
                batch = tokens[start : end + 1]
                x = batch[:-1].reshape((batch_size, seq_length))
                y = batch[1:].reshape((batch_size, seq_length))
                yield x, y

    return generator(), num_batches


@eqx.filter_jit
def loss_fn(model, xs, ys, key):
    keys = jr.split(key, xs.shape[0])
    log_p = eqx.filter_vmap(model)(xs, keys)
    log_p = jnp.take_along_axis(log_p, jnp.expand_dims(ys, 2), axis=2)
    return -jnp.mean(log_p)


def grad_loss(model, xs, ys, key):
    return eqx.filter_value_and_grad(loss_fn)(model, xs, ys, key)


@eqx.filter_jit
def make_step(model, xs, ys, optim, opt_state, key):
    key, _key = jr.split(key)
    loss, grads = grad_loss(model, xs, ys, _key)
    updates, opt_state = optim.update(
        grads, opt_state, eqx.filter(model, eqx.is_inexact_array), value=loss
    )
    model = eqx.apply_updates(model, updates)
    return model, loss, opt_state, key


def get_word_tokens(dataset, max_words=None, min_freq=0, special=["<UNK>"]):
    counter = Counter()
    for text in dataset["train"]["text"]:  # pyright: ignore
        words = text.split()
        counter.update(words)

    idx2sym = []
    sym2idx = {}

    for sym in special:
        if sym not in sym2idx:
            idx2sym.append(sym)
            sym2idx[sym] = len(idx2sym) - 1

    unk_idx = sym2idx["<UNK>"]
    for sym, cnt in counter.most_common(max_words):
        if cnt < min_freq:
            break
        if sym not in sym2idx:
            idx2sym.append(sym)
            sym2idx[sym] = len(idx2sym) - 1

    print(f"final vocab size {len(sym2idx)} from {len(counter)} unique tokens")

    def tokenize_split(split_name):
        tokens = []
        for text in dataset[split_name]["text"]:  # pyright: ignore
            words = text.split()
            token_ids = [sym2idx.get(word, unk_idx) for word in words]
            tokens.extend(token_ids)
        return np.array(tokens, dtype=np.uint32)

    train_tokens = tokenize_split("train")
    val_tokens = tokenize_split("validation")
    test_tokens = tokenize_split("test")
    vocab = sym2idx

    return train_tokens, val_tokens, test_tokens, vocab


if __name__ == "__main__":
    config = {
        "seed": 5678,
        "max_words": 50303,
        "min_freq": 0,
        "seq_length": 512,
        "embedding_size": 1024,
        "n_heads": 8,
        "dropout_p": 0.1,
        "beta": 0.5,
        "tol": 1e-3,
        "max_steps": 4,
        "epochs": 40,
        "lr": 1e-3,
        "warmup_steps": 4000,
        "weight_decay": 0.1,
        "batch_size": 32,
    }
    run = wandb.init(project="Reversible DEQ", config=config)

    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

    train_tokens, val_tokens, test_tokens, vocab = get_word_tokens(
        dataset, max_words=config["max_words"], min_freq=config["min_freq"]
    )

    train_size, val_size, test_size = (
        len(train_tokens),
        len(val_tokens),
        len(test_tokens),
    )

    vocab_size = len(vocab)
    seq_length = config["seq_length"]
    embedding_size = config["embedding_size"]
    n_heads = config["n_heads"]
    dropout_p = config["dropout_p"]
    solver = rdeq.Reversible(config["beta"])
    adjoint = rdeq.ReversibleAdjoint()
    tol = config["tol"]
    max_steps = config["max_steps"]
    key = jr.PRNGKey(config["seed"])
    model_key, data_key, training_key = jr.split(key, 3)
    model = Model(
        vocab_size,
        seq_length,
        embedding_size,
        n_heads,
        dropout_p,
        solver,
        adjoint,
        tol,
        max_steps,
        key,
    )

    batch_size = config["batch_size"]
    tokens_per_batch = batch_size * seq_length

    loader_keys = jr.split(data_key, 3)
    train_loader, train_steps_per_epoch = dataloader(
        train_tokens, batch_size, seq_length, loader_keys[0]
    )
    val_loader, val_steps_per_epoch = dataloader(
        val_tokens, batch_size, seq_length, loader_keys[1]
    )
    test_loader, test_steps_per_epoch = dataloader(
        test_tokens, batch_size, seq_length, loader_keys[2]
    )

    total_steps = train_steps_per_epoch * config["epochs"]
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config["lr"],
        warmup_steps=int(config["warmup_steps"]),
        decay_steps=total_steps,
        end_value=1e-5,
    )
    optim = optax.adamw(lr_schedule, weight_decay=config["weight_decay"])
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    epochs = int(config["epochs"])
    print_every = 100
    tokens_processed = print_every * tokens_per_batch
    global_step = 0
    tic = time.time()
    for epoch in range(epochs):
        for step, (x, y) in zip(range(train_steps_per_epoch), train_loader):
            model, loss, opt_state, training_key = make_step(
                model, x, y, optim, opt_state, training_key
            )
            global_step += 1
            if step % print_every == 0:
                toc = time.time()
                dt = toc - tic
                x, y = next(iter(val_loader))
                inference_model = eqx.nn.inference_mode(model)
                val_loss = loss_fn(inference_model, x, y, training_key)
                perplexity = jnp.exp(loss)
                val_perplexity = jnp.exp(val_loss)
                learning_rate = lr_schedule(global_step)

                run.log(
                    {
                        "step": step,
                        "dt": dt,
                        "loss": loss,
                        "perplexity": perplexity,
                        "val loss": val_loss,
                        "val perplexity": val_perplexity,
                        "lr": learning_rate,
                    }
                )
                print(
                    f"step: {step} | loss: {loss:.5f} | perplexity: {perplexity:.5f} | val loss: {val_loss:.5f} | val perplexity: {val_perplexity:.5f} | tokens/s: {(tokens_processed / dt):.0f}"
                )
                tic = time.time()

        test_loss = 0
        count = 0
        for _, (x, y) in zip(range(test_steps_per_epoch), test_loader):
            inference_model = eqx.nn.inference_mode(model)
            test_loss += loss_fn(inference_model, x, y, training_key)
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
