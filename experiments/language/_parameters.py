def parameters(seq_length, vocab_size, embedding_size, n_heads, n_blocks):
    pos_emb = seq_length * embedding_size
    tok_emb = vocab_size * embedding_size

    # Block
    attention = 4 * n_heads * embedding_size**2
    mlp = 8 * embedding_size**2

    # Linear
    head = embedding_size * vocab_size

    params = pos_emb + tok_emb + n_blocks * (attention + mlp) + head
    non_embed_size = n_blocks * (attention + mlp)

    return params, non_embed_size


if __name__ == "__main__":
    seq_length = 448
    vocab_size = 50304
    embedding_size = 1024
    n_heads = 12
    n_blocks = 1

    params, non_embed_size = parameters(
        seq_length, vocab_size, embedding_size, n_heads, n_blocks
    )

    print(
        f"Parameters: {params / 1e6:.0f}M, Non-Embedding Size: {non_embed_size / 1e6:.0f}M"
    )
