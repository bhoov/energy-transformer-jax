# Energy Transformer

> A novel architecture that is a Transformer, an Energy-Based Model, and an Associative Memory. See [our paper](https://arxiv.org/abs/2302.07253)

## Structure

This repository has been cleaned and rewritten for the purpose of clear communication rather than complete features (as was done in the experiments of the paper). The architecture is built using [`equinox`](https://github.com/patrick-kidger/equinox), an excellent and barebones JAX library that looks a lot like pytorch. All pseudocode examples in this README use equinox.

For legacy purposes we include the [`flax`](https://github.com/google/flax) code that was used in the original paper in the `og_implementation` folder. 

## Introduction

Energy Transformer (ET) is a continuous dynamical system with a tractable energy -- this means that the forward pass through the model can be done using autograd! 

## Quick start
We are still in the process of cleaning up the environment setup for this repository. For the main tutorial code, you can run:


Demo code will work on a CPU.


### Examples

For a complete example of ET (with randomly initialized weights), see `example.py`. Observe energy behavior by running

```
python example.py
```

## Minimal ET

TODO ET is incredibly simple. We write the pseudocode for the model definition and inference below:

``` python
import equinox as eqx
import jax
import 

class EnergyTransformer(eqx.Module):
    Wq: jax.Array
    Wk: jax.Array
    M:  jax.Array
    
    def __init__(self, key, )

```

## Testing

```
pytest tests
```

## Original Implementation
To our chagrin, the code used to run the experiments in the paper was implemented in Jupyter Notebooks and flax. We are in the process of improving cleanliness of this code. You can check out our progress in the `og_implementation/` folder.


