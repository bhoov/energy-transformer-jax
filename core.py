# %%
from typing import *
from pathlib import Path
import functools as ft
from dataclasses import dataclass
# import os
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".9"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from einops import rearrange

from architecture import (
  ETConfig,
  EnergyLayerNorm,
  EnergyAttention,
  HopfieldNetwork,
  EnergyTransformer,
)

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
    self.patch_elements = ft.reduce(lambda a, b=1: a * b, self.patch_shape)

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


@dataclass
class FullConfig:
  image_shape: Tuple[int, int, int]  # Shape of the image to be processed
  patch_size: int  # Size of the patches to extract from the image
  et_conf: ETConfig = eqx.field(static=True)  # Configuration for the EnergyTransformer
  n_mask: int = 100  # Number of patches to mask out


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

    cls_key, mask_key, pos_key, enc_key, dec_key, et_key = jr.split(key, 6)

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

    x = x[1:]  # Discard CLS token, only needed to collect global image representation
    g = self.lnorm(x)
    x = self.decode(g)
    x = self.patcher.unpatchify(x)
    return x


# %%
def load_checkpoint(path: Union[str, Path]):
  """Load trained weights into our equinox module

  A manual process because the checkpoint was saved using different code in Flax
  """
  load_dict = dict(**np.load(path))
  H, Y, D = load_dict["Wk"].shape
  D, M = load_dict["Xi"].shape
  scale_mems = M / D
  et_conf = ETConfig(D, Y, n_heads=H, scale_mems=scale_mems)
  full_conf = FullConfig(
    image_shape=(3, 224, 224),
    patch_size=16,
    et_conf=et_conf,
  )

  key = jr.PRNGKey(0)
  et = EnergyTransformer(key, et_conf)

  attn = EnergyAttention(key, et_conf)
  attn = eqx.tree_at(lambda attn: attn.Wk, attn, load_dict["Wk"])
  attn = eqx.tree_at(lambda attn: attn.Wq, attn, load_dict["Wq"])
  et = eqx.tree_at(lambda et: et.attn, et, attn)

  hn = HopfieldNetwork(key, et_conf)
  hn = eqx.tree_at(lambda hn: hn.Xi, hn, load_dict["Xi"])
  et = eqx.tree_at(lambda et: et.hn, et, hn)

  iet = ImageEnergyTransformer(key, full_conf)
  iet = eqx.tree_at(lambda iet: iet.et, iet, et)

  enc = Linear(key, dim_in=iet.patcher.patch_elements, dim_out=et_conf.D)
  enc = eqx.tree_at(lambda enc: enc.W, enc, load_dict["Wenc"])
  enc = eqx.tree_at(lambda enc: enc.bias, enc, load_dict["Benc"])
  iet = eqx.tree_at(lambda iet: iet.encoder, iet, enc)

  dec = Linear(key, dim_in=et_conf.D, dim_out=iet.patcher.patch_elements)
  dec = eqx.tree_at(lambda dec: dec.W, dec, load_dict["Wdec"])
  dec = eqx.tree_at(lambda dec: dec.bias, dec, load_dict["Bdec"])
  iet = eqx.tree_at(lambda iet: iet.decoder, iet, dec)

  lnorm = EnergyLayerNorm(D, use_bias=True)
  lnorm = eqx.tree_at(lambda lnorm: lnorm.gamma, lnorm, load_dict["LNORM_gamma"])
  lnorm = eqx.tree_at(lambda lnorm: lnorm.delta, lnorm, load_dict["LNORM_bias"])
  iet = eqx.tree_at(lambda iet: iet.lnorm, iet, lnorm)

  cls_token = load_dict["CLS_token"]
  mask_token = load_dict["MASK_token"]
  pos_embed = load_dict["POS_embed"]
  iet = eqx.tree_at(lambda iet: iet.cls_token, iet, cls_token)
  iet = eqx.tree_at(lambda iet: iet.mask_token, iet, mask_token)
  iet = eqx.tree_at(lambda iet: iet.pos_embed, iet, pos_embed)
  return iet


iet = load_checkpoint("./checkpoints/plaindict_ckpt.npz")

# %% Load image from real dataset
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255


def img_to_array(im):
  """Put into channel first format, normalize"""
  x = np.array(im)
  x = (x - IMAGENET_MEAN) / IMAGENET_STD
  x = rearrange(x, "h w c-> c h w")
  return x


def array_to_img(x):
  """Put back into channel last format, denormalize"""
  x = rearrange(x, "c h w -> h w c")
  x = (x * IMAGENET_STD) + IMAGENET_MEAN
  return x


if __name__ == "__main__":
  """Run the following test code to see if the model can be loaded and run on a single image"""
  # %%
  from PIL import Image
  import matplotlib.pyplot as plt

  iet = load_checkpoint("./checkpoints/plaindict_ckpt.npz")

  im = Image.open("imgs/00_parrot.png").convert("RGB")
  x = jnp.array(img_to_array(im))
  mask = np.zeros((iet.patcher.n_patches,), dtype=np.uint8)
  key = jr.PRNGKey(0)
  mask_idxs = jr.choice(
    key, np.arange(iet.patcher.n_patches), shape=(iet.conf.n_mask,), replace=False
  )
  # mask_idxs = np.random.choice(np.arange(iet.patcher.n_patches), size=iet.patcher.n_patches // 2, replace=False)
  mask[mask_idxs] = 1
  mask = jnp.array(mask, dtype=jnp.uint8)

  # masked input
  xin = iet.patcher.patchify(x)
  xin = iet.encode(xin)
  xin = iet.prep_tokens(xin, mask)  # N+1,D
  masked_input = iet.patcher.unpatchify(iet.decode(iet.lnorm(xin[1:])))
  og_input = np.array(im)

  out = iet(x, mask)

  fig, axs = plt.subplots(1, 3, figsize=(12, 4))
  ax = axs[0]
  ax.imshow(array_to_img(masked_input) / 255.)
  ax.set_title("Masked Input")
  ax.set_xticks([])
  ax.set_yticks([])

  ax = axs[1]
  ax.imshow(array_to_img(out) / 255.)
  ax.set_title("Reconstruction")
  ax.set_xticks([])
  ax.set_yticks([])
  
  ax = axs[2]
  ax.imshow(og_input / 255.)
  ax.set_title("Original Image")
  ax.set_xticks([])
  ax.set_yticks([])