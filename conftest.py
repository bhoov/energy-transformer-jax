import pytest
from example import ETConfig, EnergyLayerNorm
import jax
import jax.numpy as jnp
import jax.random as jr

@pytest.fixture
def _etconf():
  return ETConfig(
    D=12,
    Y=8,
    n_heads=2,
    scale_mems=2
  )

@pytest.fixture
def _lnorm(_etconf):
  return EnergyLayerNorm(_etconf.D)
