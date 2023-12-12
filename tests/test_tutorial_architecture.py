import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

def test_lnorm(_lnorm, _etconf):
  ln = _lnorm
  x = 23 * jr.normal(jr.PRNGKey(0), (_etconf.D,)) - 1
  auto_g = jax.grad(ln.lagrangian)(x)
  man_g = ln.g(x)
  assert jnp.allclose(auto_g, man_g)