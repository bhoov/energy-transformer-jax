import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from dataclasses import dataclass
from typing import *

@dataclass
class ETConfig():
  D: int = 768 # Token dimension of ET
  Y: int = 64 # Token dimension of each query and key
  n_heads: int = 12 # Number of heads
  scale_mems: float = 4. # Scale the number of memories by this factor relative to token dimension D

class EnergyLayerNorm(eqx.Module):
  """Define our primary activation function (modified LayerNorm) as a lagrangian with energy"""
  gamma: jax.Array  # Scaling scalar
  delta: jax.Array  # Bias
  use_bias: bool
  eps: float
  
  def __init__(self, dim: int, use_bias:bool = True, eps:float = 1e-5):
    self.use_bias = use_bias
    self.gamma = jnp.ones(())
    self.delta = jnp.zeros(dim)
    self.eps = eps
    
  def lagrangian(self, x):
    """The integral of the standard LayerNorm, with the following twist: `gamma` is a scalar, not a vector of shape `dim` as in the original layernorm """
    D = x.shape[-1]
    xmeaned = x - x.mean(-1, keepdims=True)
    t1 = D * self.gamma * jnp.sqrt((1 / D * xmeaned**2).sum() + self.eps)
    if not self.use_bias: 
      return t1
    t2 = (self.delta * x).sum()
    return t1 + t2

  def g(self, x):
    """The manual derivative of the lagrangian. 
    
    You could compute this with autograd, but it is more efficient and clear to implement it directly
    """
    xmeaned = x - x.mean(-1, keepdims=True)
    v = self.gamma * (xmeaned) / jnp.sqrt((xmeaned**2).mean(-1, keepdims=True)+ self.eps)
    if self.use_bias:
        return v + self.delta
    return v

  def __call__(self, x):
    """An alias for the activation function `g`"""
    return self.g(x)
    
  def energy(self, x):
    """Compute the energy of this Lagrangian through the Legendre Transform"""
    return (self.g(x) * x).sum() - self.lagrangian(x)
    
class EnergyAttention(eqx.Module):
  """Our novel attention with energy

  Has only two learnable parameters, Wk and Wq
  """
  Wq: jax.Array
  Wk: jax.Array
  config: ETConfig = eqx.field(static=True)

  def __init__(self, key:jr.PRNGKey, config:ETConfig):
    kkey, qkey = jr.split(key)
    self.Wk = jr.normal(kkey, (config.n_heads, config.Y, config.D))
    self.Wq = jr.normal(qkey, (config.n_heads, config.Y, config.D))
    self.config = config

  def energy(self, g:jnp.ndarray):
    """Return the energy of the block. The update rule is autograd through this function"""
    beta = 1/jnp.sqrt(self.config.Y)
    K = jnp.einsum("kd,hzd->khz", g, self.Wk) # nKeys,nHeads,Y
    Q = jnp.einsum("qd,hzd->qhz", g, self.Wq) # nQueries,nHeads,Y
    A = jax.nn.logsumexp(beta * jnp.einsum("qhz,khz->hqk", Q, K), -1) # nHeads,nQueries,nKeys
    return -1/beta * A.sum()

class HopfieldNetwork(eqx.Module):
  """ A simple Hopfield Network (we use ReLU as the activation function) replaces the MLP in traditional Transformers """
  Xi: jax.Array

  def __init__(self, key:jr.PRNGKey, config:ETConfig):
    nmems = int(config.scale_mems * config.D)
    self.Xi = jr.normal(key, (nmems, config.D))

  def energy(self, g:jnp.ndarray):
    """Return the Hopfield Network's energy"""
    hid = jnp.einsum("nd,md->nm", g, self.Xi) # nTokens, nMems
    E = -0.5 * (jax.nn.relu(hid) ** 2).sum()
    return E

class EnergyTransformer(eqx.Module):
  """A simple wrapper class that sums the energies of the Hopfield Network and the Attention"""
  attn: EnergyAttention
  hn: HopfieldNetwork
  config: ETConfig = eqx.field(static=True)
  
  def __init__(self, key:jr.PRNGKey, config:ETConfig):
    attn_key, hn_key = jr.split(key)
    self.attn = EnergyAttention(attn_key, config)
    self.hn = HopfieldNetwork(hn_key, config)
    self.config = config

  def energy(self, g:jnp.ndarray):
    """Return the energy of the whole Transformer"""
    return self.attn.energy(g) + self.hn.energy(g)
  