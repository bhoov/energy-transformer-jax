"""
A minimal working example of Energy Transformer written in JAX and Equinox
"""
#%% 
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import os
import equinox as eqx
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".9"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

from dataclasses import dataclass
from typing import *

@dataclass
class ETConfig():
  D: int = 768 # Token dimension of ET
  Y: int = 64 # Token dimension of each query and key
  n_heads: int = 12 # Number of heads
  scale_mems: float = 4. # Scale the number of memories by this factor relative to token dimension D

class EnergyAttention(eqx.Module):
  Wq: jax.Array
  Wk: jax.Array
  config: ETConfig

  def __init__(self, key:jr.PRNGKey, config:ETConfig):
    kkey, qkey = jr.split(key)
    self.Wk = jr.normal(kkey, (config.n_heads, config.Y, config.D))
    self.Wq = jr.normal(qkey, (config.n_heads, config.Y, config.D))

  def energy(self, g:jnp.ndarray):
    """Return the energy of the block"""
    beta = 1/jnp.sqrt(self.config.Y)
    K = jnp.einsum("kd,hzd->khz", g, self.Wk) # nKeys,nHeads,Y
    Q = jnp.einsum("qd,hzd->qhz", g, self.Wq) # nQueries,nHeads,Y
    A = jax.nn.logsumexp(beta * jnp.einsum("qhz,khz->hqk"), -1) # nHeads,nQueries,nKeys
    return -1/beta * A.sum()

class HopfieldNetwork(eqx.Module):
  Xi: jax.Array
  config: ETConfig

  def __init__(self, key:jr.PRNGKey, config:ETConfig):
    nmems = int(config.scale_mems * config.D)
    self.Xi = jr.normal(key, (nmems, config.D))

  def energy(self, g:jnp.ndarray):
    """Return the energy of the block"""
    hid = jnp.einsum("nd,md->nm", g, self.Xi) # nTokens, nMems
    E = -0.5 * (jax.nn.relu(hid) ** 2).sum()
    return E

class EnergyTransformer(eqx.Module):
  attn: EnergyAttention
  hn: HopfieldNetwork
  config: ETConfig
  
  def __init__(self, key:jr.PRNGKey, config:ETConfig):
    attn_key, hn_key = jr.split(key)
    self.attn = EnergyAttention(attn_key, config)
    self.hn = HopfieldNetwork(hn_key, config)
    self.config = config

  def energy(self, g:jnp.ndarray):
    """Return the energy of the block"""
    return self.attn.energy(g) + self.hn.energy(g)
    

  
# %%
