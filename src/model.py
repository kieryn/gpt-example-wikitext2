"""
Transformer decoder-only model in Flax.
"""

import jax.numpy as jnp
import flax.linen as nn

class TransformerDecoder(nn.Module):
    vocab_size: int
    d_model: int
    num_heads: int
    num_layers: int
    max_seq_length: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, input_ids, train: bool = True):
        x = nn.Embed(self.vocab_size, self.d_model)(input_ids)
        pos_emb = self.param('pos_embedding', nn.initializers.normal(stddev=0.02), (1, self.max_seq_length, self.d_model))
        x = x + pos_emb[:, :x.shape[1], :]

        for _ in range(self.num_layers):
            x = nn.LayerNorm()(x)
            x = nn.SelfAttention(
                num_heads=self.num_heads,
                qkv_features=self.d_model,
                dropout_rate=self.dropout_rate,
                deterministic=not train
            )(x)
            x = nn.Dense(self.d_model)(x)

        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.vocab_size)(x)
        return logits
