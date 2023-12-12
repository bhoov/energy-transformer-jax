#%%
"""
A minimal working example of Energy Transformer written in JAX and Equinox
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from dataclasses import dataclass
from typing import *
from architecture import ETConfig, EnergyLayerNorm, EnergyAttention, HopfieldNetwork, EnergyTransformer

## Adjust the following env variables if you intend to use a GPU
# import os
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".9"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

if __name__ == "__main__":
  """ Only run the following code in interactive mode (VSCode) or as a script """
  #%%
  import matplotlib.pyplot as plt

  config = ETConfig()
  n_tokens = 11
  rng = jr.PRNGKey(0)
  et = EnergyTransformer(rng, config)
  lnorm = EnergyLayerNorm(config.D, use_bias=False)

  def energy(x):
    E1 = lnorm.energy(x)
    g = lnorm.g(x)
    E2 = et.energy(g)
    return E1 + E2

  key, rng = jr.split(rng)
  x = jr.normal(key, (n_tokens, config.D))
  energies = []
  nsteps = 150
  alpha = 0.1 # Stepsize

  get_energy = jax.jit(jax.value_and_grad(energy))
  xmins = []
  xmaxes = []
  gdiffs = []

  gprev = None
  for i in range(nsteps):
    g = lnorm(x)
    if gprev is not None:
      gdiff = jnp.max(jnp.abs(g - gprev))
      gdiffs.append(gdiff)
      
    gprev = g
    E, dEdg = get_energy(g)
    x = x - alpha * dEdg
    xmaxes.append(jnp.max(x))
    xmins.append(jnp.min(x))
    energies.append(E)

  plt.figure()
  plt.plot(energies) 
  plt.xlabel("Time (number of iterations)")
  plt.ylabel("Energy")
  plt.title("ET's energy is a fixed point attractor")
    
  fig, axs = plt.subplots(3,1)
  ax = axs[0]
  ax.plot(gdiffs) 
  ax.set_title("$\Delta$ g")
  ax = axs[1]
  ax.plot(xmaxes) 
  ax.set_title("Xmax")
  ax = axs[2]
  ax.plot(xmins) 
  ax.set_title("Xmin")

  plt.show()