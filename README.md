# Energy Transformer

> A novel architecture that is a Transformer, an Energy-Based Model, and an Associative Memory. See [our paper](https://arxiv.org/abs/2302.07253)

## Structure

This repository has been cleaned and rewritten for the purpose of clear communication rather than complete features (as was done in the experiments of the paper). The architecture is built using [`equinox`](https://github.com/patrick-kidger/equinox), an excellent and barebones JAX library that looks a lot like pytorch. All pseudocode examples in this README use equinox.

For legacy purposes we include the [`flax`](https://github.com/google/flax) code that was used in the original paper in the `og_implementation` folder. 

## Introduction

Energy Transformer (ET) is a continuous dynamical system with a tractable energy -- this means that the forward pass through the model can be done using autograd! This comes with additional benefits like being highly parameter efficient and interpretable (TODO add links). **Pseudocode** on layernorm representations `g` below:

``` python
import equinox as eqx
import jax
class EnergyTransformer(eqx.Module):
    # Define all parameters
    Wq: jax.Array  # n_heads, head_dim, token_dim
    Wk: jax.Array  # n_heads, head_dim, token_dim
    Xi: jax.Array  # n_memories, token_dim

    def __init__(self, token_dim, n_heads, head_dim, n_memories):
        ...

    def attn_energy(self, g):
        Q = jnp.einsum("qd,hzd->qhz", g, self.Wq)
        K = jnp.einsum("kd,hzd->khz", g, self.Wk)

        beta = 1 / jnp.sqrt(head_dim)
        A = -1 / beta * jax.nn.logsumexp(beta * jnp.einsum("qhz,khz->hqk", Q, K), -1).sum()
    
    def hn_energy(self, g):
        return -1 / 2 * jax.nn.relu(jnp.einsum("nd,md->nm", g, self.Xi)).sum()

    def energy(self, g):
        return self.attn_energy(g) + self.hn_energy(g)

et = EnergyTransformer(...)

key = jr.PRNGkey(0)
x = jr.normal(key, (n_tokens, token_dim))

for i in range(n_steps):
    g = lnorm(x)
    E, dEdg = jax.value_and_grad(et.energy)(g)
    x = x - alpha * dEdg
```

There is also an energy on the LayerNorm that we cannot ignore, but the above is an excellent starting point for the architecture. See working code in `tutorial.py.`

## Quick start
We are still in the process of cleaning up the environment setup for this repository. For the main tutorial code, you can run:

```
conda env create -f environment.yml
conda activate et-jax
pip install -r requirements.txt
```

Demo code (randomized weights) and environment works on a CPU. Observe energy behavior:

```
python tutorial.py
```

## Testing

```
pytest tests
```

## Original Implementation
To our chagrin, the code used to run the experiments in the paper is incredibly messy and not good for understanding. We are in the process of improving cleanliness of this code. You can check out our progress in the `og_implementation/` folder.