#%%
%reload_ext autoreload
%autoreload 2

# %%
import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import equinox as eqx
from typing import *
from pathlib import Path
import functools as ft
from einops import rearrange
from dataclasses import dataclass

# import os
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".9"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

from architecture import (
  ETConfig,
  EnergyLayerNorm,
  EnergyAttention,
  HopfieldNetwork,
  EnergyTransformer,
)


def ifnone(a, b):
  return b if a is None else a


def is_squarable(n: int):
  """Check if 2Darray `x` is square"""
  m = int(np.sqrt(n))
  return m == np.sqrt(n)


def mul(a, b=1):
  return a * b


def normal(key, shape, mean=0, std=1):
  x = jax.random.normal(key, shape)
  return (x * std) + mean


class Patcher:
  image_shape: Iterable[int]
  patch_size: int
  kh: int
  kw: int

  def __init__(self, image_shape, patch_size, kh, kw):
    """Class with a `patchify` and `unpatchify` method for a provided image shape.

    **Assumes "CHW" image ordering**

    Args:
        kh: number of patches in the height direction
        kw: number of patches in the width direction
        has_batch: If True, treat all provided images as having a batch dimension

    Usage:
        patcher = Patcher.from_img_shape((3,32,32), 8, has_batch=False)
        # x = single img from CIFAR
        y = patcher.patchify(x) # y.shape = (n_patches, C, patch_size, patch_size)
        z = patcher.unpatchify(y) # z.shape = (C, H, W)
    """
    self.image_shape = image_shape
    self.patch_size = patch_size
    self.kh, self.kw = kh, kw
    self.n_patches = kh * kw
    self.patch_shape = (image_shape[0], patch_size, patch_size)
    self.patch_elements = ft.reduce(mul, self.patch_shape)

  def patchify(self, img):
    return rearrange(
      img,
      "... c (kh h) (kw w)-> ... (kh kw) c h w",
      h=self.patch_size,
      w=self.patch_size,
    )

  def unpatchify(self, patches):
    return rearrange(
      patches, "... (kh kw) c h w -> ... c (kh h) (kw w)", kh=self.kh, kw=self.kw
    )

  def patchified_shape(self):
    """Return the expected size output from the data generator"""
    return (self.n_patches, *self.patch_shape)

  @classmethod
  def from_img(cls, img, patch_size):
    return cls.from_img_shape(img.shape, patch_size)

  @classmethod
  def from_img_shape(cls, img_shape, patch_size):
    height, width = img_shape[-2:]
    assert (height % patch_size) == 0
    assert (width % patch_size) == 0
    kh = int(height / patch_size)
    kw = int(width / patch_size)

    return cls(img_shape, patch_size, kh, kw)


class Linear(eqx.Module):
  W: jax.Array
  bias: Optional[jax.Array]

  def __init__(self, key: jax.Array, dim_in: int, dim_out: int, use_bias=True):
    self.W = 0.1 * jr.normal(key, (dim_in, dim_out))
    if use_bias:
      self.bias = jnp.zeros((dim_out,))
    else:
      self.bias = None

  def __call__(self, x):
    if self.bias is None:
      return x @ self.W
    return x @ self.W + self.bias

#%%
@dataclass
class FullConfig():
  image_shape: Tuple[int, int, int] # Shape of the image to be processed
  patch_size: int # Size of the patches to extract from the image
  et_conf: ETConfig = eqx.field(static=True)# Configuration for the EnergyTransformer
  n_mask: int = 100 # Number of patches to mask out


class ImageEnergyTransformer(eqx.Module):
  patcher: Patcher
  encoder: Linear
  decoder: Linear
  cls_token: jax.Array
  mask_token: jax.Array
  pos_embed: jax.Array
  et: EnergyTransformer
  lnorm: EnergyLayerNorm
  conf: FullConfig

  def __init__(self, key: jax.Array, conf: FullConfig):
    self.conf = conf
    self.patcher = Patcher.from_img_shape(conf.image_shape, conf.patch_size)

    cls_key, mask_key, pos_key, enc_key, dec_key, et_key  = jr.split(key, 6)

    self.cls_token = 0.002 * jr.normal(cls_key, (conf.et_conf.D,))
    self.mask_token = 0.002 * jr.normal(mask_key, (conf.et_conf.D,))
    self.pos_embed = 0.002 * jr.normal(
      pos_key, (1 + self.patcher.n_patches, conf.et_conf.D)
    )

    self.encoder = Linear(
      enc_key, dim_in=self.patcher.patch_elements, dim_out=conf.et_conf.D
    )
    self.decoder = Linear(
      dec_key, dim_in=conf.et_conf.D, dim_out=self.patcher.patch_elements
    )
    self.et = EnergyTransformer(et_key, conf.et_conf)
    self.lnorm = EnergyLayerNorm(conf.et_conf.D)

  def encode(self, x):
    """Turn x from img patches to tokens"""
    x = rearrange(x, "... c h w -> ... (c h w)")
    x = self.encoder(x)
    return x

  def decode(self, x):
    g = self.lnorm(x)
    x = self.decoder(g)
    c, h, w = self.patcher.patch_shape
    return rearrange(x, "... (c h w) -> ... c h w", c=c, h=h, w=w)

  def corrupt_tokens(
    self,
    x: jax.Array,  # The input tokens of shape ND
    mask: jax.Array,  # Mask (uint8) of shape N.
  ) -> jax.Array:
    """Corrupt tokens `x` according to schema provided in `mask`."""
    maskmask = jnp.nonzero(mask == 1, size=self.conf.n_mask, fill_value=0)
    x = x.at[maskmask].set(self.mask_token)
    return x

  def prep_tokens(
    self,
    x: jax.Array,  # Encoded tokens of shape ND
    mask: jax.Array,  # uint8 mask of shape N
  ) -> jax.Array:  # Tokens of shape B(N+1)D where tokens have been appropriately masked
    """Prepare tokens for masked image modeling, adding CLS,MASK tokens and POS embeddings"""
    x = self.corrupt_tokens(x, mask)
    x = jnp.concatenate([self.cls_token[None], x])
    x = x + self.pos_embed
    return x

  def __call__(self, x: jnp.ndarray, mask: jnp.ndarray, nsteps=12, alpha=0.1):
    """A complete pipeline for masked image modeling in ET"""
    x = self.patcher.patchify(x)
    x = self.encode(x)
    x = self.prep_tokens(x, mask)  # N+1,D

    get_energy_info = jax.value_and_grad(self.et.energy)
    for i in range(nsteps):
      g = self.lnorm(x)
      E, dEdg = get_energy_info(g)
      x = x - alpha * dEdg

    x = x[1:] # Discard CLS token, only needed to collect global image representation
    g = self.lnorm(x)
    x = self.decode(g)
    x = self.patcher.unpatchify(x)
    return x

# %%
conf = FullConfig(
  image_shape=(3, 224, 224),
  patch_size=16,
  et_conf=ETConfig(D=768, Y=64, n_heads=4, scale_mems=4),
)
rng = jr.PRNGKey(0)
iet_key, rng = jr.split(rng)
iet = ImageEnergyTransformer(jr.PRNGKey(0), conf)

img = jr.normal(jr.PRNGKey(0), (conf.image_shape))

mask = np.zeros((iet.patcher.n_patches,), dtype=np.uint8)
mask_idxs = np.random.choice(np.arange(iet.patcher.n_patches), size=iet.patcher.n_patches // 2, replace=False)
mask[mask_idxs] = 1
mask = jnp.array(mask, dtype=jnp.uint8)

out = iet(img, mask)

#%%
import matplotlib.pyplot as plt
plt.imshow(rearrange(out, "c h w -> h w c"))

#%%
mask = np.random.randint(0, 2, size=iet.patcher.n_patches)
#%%
def load_checkpoint(path: Union[str, Path]):
  """Load checkpoints saved using Flax into our equinox module"""
  load_dict = dict(**np.load(path))
  H, Y, D = load_dict["Wk"].shape
  D, M = load_dict["Xi"].shape
  scale_mems = M / D
  lnorm = EnergyLayerNorm(D, use_bias=True)
  lnorm = eqx.tree_at(lambda lnorm: lnorm.gamma, lnorm, load_dict["LNORM_gamma"])
  lnorm = eqx.tree_at(lambda lnorm: lnorm.delta, lnorm, load_dict["LNORM_bias"])

  et_conf = ETConfig(D, Y, n_heads=H, scale_mems=scale_mems)
  full_conf = FullConfig(
    image_shape=(3, 224, 224),
    patch_size=16,
    et_conf=et_conf,
  )
  rng = jr.PRNGKey(0)
  iet_key, rng = jr.split(rng)
  iet = ImageEnergyTransformer(iet_key, full_conf)

  attn = EnergyAttention(jr.PRNGKey(0), et_conf)
  attn = eqx.tree_at(lambda attn: attn.Wk, attn, load_dict["Wk"])
  attn = eqx.tree_at(lambda attn: attn.Wq, attn, load_dict["Wq"])

  hn = HopfieldNetwork(jr.PRNGKey(0), et_conf)
  hn = eqx.tree_at(lambda hn: hn.Xi, hn, load_dict["Xi"])

  EnergyTransformer = eqx.tree_at(lambda et: et.norm, EnergyTransformer, lnorm)

  patch_size = 3 * 16 * 16
  enc = Linear(jr.PRNGKey(0), dim_in=patch_size, dim_out=et_conf.D)
  enc = eqx.tree_at(lambda enc: enc.W, enc, load_dict["Wenc"])
  enc = eqx.tree_at(lambda enc: enc.bias, enc, load_dict["Benc"])

  dec = Linear(jr.PRNGKey(0), dim_in=et_conf.D, dim_out=patch_size)
  dec = eqx.tree_at(lambda dec: dec.W, dec, load_dict["Wdec"])
  dec = eqx.tree_at(lambda dec: dec.bias, dec, load_dict["Bdec"])

  return load_dict, enc, dec


load_dict, enc, dec = load_checkpoint("./checkpoints/plaindict_ckpt.npz")
# load_dict = load_checkpoint('./checkpoints/plaindict_ckpt.npz')
# print(load_dict['LNORM_gamma'].shape)

# EnergyLayerNorm()

# %%
load_dict.keys()
# %%
