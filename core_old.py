#%%
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import flax.linen as nn
from typing import *
from pathlib import Path
from typing import Union
# import cloudpickle
import functools as ft
from einops import rearrange
from pathlib import Path
from enum import IntEnum, auto

#%% Architectures
class MultiheadAttention(nn.Module):
    """The energy of attention for a single head"""
    tokdim: int=768
    nheads: int = 12
    kspace_dim: int = 64
    use_bias: bool = False
    param_dtype: Any = jnp.float32
    beta_init: Optional[float] = None

    def setup(self):
        self.Wk = self.param("Wk", nn.initializers.normal(0.002), (self.nheads, self.kspace_dim, self.tokdim), self.param_dtype)
        self.Wq = self.param("Wq", nn.initializers.normal(0.002), (self.nheads, self.kspace_dim, self.tokdim), self.param_dtype)
        self.betas = jnp.ones(self.nheads, dtype=self.param_dtype) * ifnone(self.beta_init, 1/jnp.sqrt(self.kspace_dim))

        if self.use_bias:
            bq = self.param("bq", self.bias_init, (self.nheads, self.kspace_dim), self.param_dtype)
            bk = self.param("bk", self.bias_init, (self.nheads, self.kspace_dim), self.param_dtype)

    def energy(self, g:jnp.ndarray):
        """Return the energy of the block"""
        K = jnp.einsum("kd,hzd->khz", g, self.Wk) # kseq,heads,kspace
        Q = jnp.einsum("qd,hzd->qhz", g, self.Wq) # qseq,heads,kspace
        if self.use_bias:
            K = K + self.bk
            Q = Q + self.bq

        A1 = jnp.einsum("h,qhz,khz->hqk", self.betas, Q, K) # heads, qseq, kseq
        A2 = jax.nn.logsumexp(A1, -1) # heads, qseq
        A3 = A2.sum(-1) # heads
        A4 = (-1/self.betas * A3).sum()
        return A4

    def energy_and_grad(self, g:jnp.ndarray):
        return jax.value_and_grad(self.energy)(g)

    def manual_grad(self, g:jnp.ndarray) -> jnp.ndarray:
        K = jnp.einsum("hzd,kd->khz", self.Wk, g)
        Q = jnp.einsum("hzd,qd->qhz", self.Wq, g)
        F1 = jnp.einsum("hzd,khz->khd", self.Wq, K)
        F2 = jnp.einsum("hzd,qhz->qhd", self.Wk, Q)

        A1 = jnp.einsum("h,khz,qhz->hqk", self.betas, K, Q)
        A2 = jax.nn.softmax(A1, -1) # hqk

        T1 = -jnp.einsum("khd,hqk->qd", F1, A2)
        T2 = -jnp.einsum("qhd,hqk->kd", F2, A2)
        return T1 + T2

class CHNReLU(nn.Module):
    tokdim: int
    hidden_ratio: float = 4.
    param_dtype:Any = jnp.float32
    use_bias: bool=False

    def setup(self):
        hid_dim = int(self.hidden_ratio * self.tokdim)
        self.kernel = self.param("kernel", nn.initializers.normal(0.02), (self.tokdim, hid_dim), self.param_dtype)
        if self.use_bias:
            self.bias = self.param("bias", nn.initializers.zeros, (hid_dim,), self.param_dtype)

    def energy(self, g:jnp.ndarray):
        h = g @ self.kernel
        if self.use_bias:
            h += self.bias
        A = jax.nn.relu(h)
        return -0.5*(A**2).sum()

    def energy_and_grad(self, g:jnp.ndarray):
        return jax.value_and_grad(self.energy)(g)

    def manual_grad(self, g: jnp.ndarray):
        """Only used for checking the automatic gradient calculation"""
        h = g @ self.kernel
        if self.use_bias:
            h += self.bias
        A1 = jax.nn.relu(h) # hid
        A2 = -A1 @ self.kernel.T # D
        return A2

class CHNSoftmax(nn.Module):
    tokdim: int
    hidden_ratio: float = 4.
    param_dtype: Any = jnp.float32
    use_bias: bool=False
    beta_init: float = 0.01

    def setup(self):
        hid_dim = int(self.hidden_ratio * self.tokdim)
        self.kernel = self.param("kernel", nn.initializers.normal(0.02), (self.tokdim, hid_dim), self.param_dtype)
        self.beta = self.param(
            "beta",
            lambda key, shape, dtype: nn.initializers.ones(key, shape, dtype)*self.beta_init,
            (), self.param_dtype)
        if self.use_bias:
            self.bias = self.param("bias", nn.initializers.zeros, (hid_dim,), self.param_dtype)

    def energy(self, g:jnp.ndarray):
        h = self.beta * g @ self.kernel
        if self.use_bias:
            h = h + self.bias
        A = jax.nn.logsumexp(h, axis=-1) # hid
        return -1/self.beta * A.sum()

    def energy_and_grad(self, g:jnp.ndarray):
        return jax.value_and_grad(self.energy)(g)

    def manual_grad(self, g:jnp.ndarray):
        """Only used for checking the automatic gradient calculation"""
        h = self.beta * g @ self.kernel
        if self.use_bias:
            h = h + self.bias
        A1 = jax.nn.softmax(h, axis=-1) # hid
        A2 = -A1 @ self.kernel.T # D
        return A2

class EnergyLayerNorm(nn.Module):
    """Do layer norm on the last dimension of input

    While an energy could be defined for this, it is easier to just define the forward operation (activation function) since the
    energy calculation is not needed for the dynamics of the network
    """
    dim: int
    param_dtype: Any = jnp.float32
    use_bias:bool = True # Whether to use a bias in the layer normalization step or not
    eps: float = 1e-05 # Prevent division by 0

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        gamma = self.param("gamma", nn.initializers.ones, (), self.param_dtype)
        if self.use_bias:
            bias = self.param ("bias", nn.initializers.zeros, (self.dim,), self.param_dtype)
        xmeaned = x - x.mean(-1, keepdims=True)
        v = gamma * (xmeaned) / jnp.sqrt((xmeaned**2).mean(-1, keepdims=True)+self.eps)
        if self.use_bias:
            return v + bias
        return v

class RegularizedEnergyLayerNorm(nn.Module):
    """Do layer norm on the last dimension of input

    While an energy could be defined for this, it is easier to just define the forward operation (activation function) since the
    energy calculation is not needed for the dynamics of the network
    """
    dim: int
    param_dtype: Any = jnp.float32
    use_bias:bool = True # Whether to use a bias in the layer normalization step or not
    eps: float = 1e-05 # Prevent division by 0
    lmda: float = 1e-03 # Regularization

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        gamma = self.param("gamma", nn.initializers.ones, (), self.param_dtype)
        if self.use_bias:
            bias = self.param ("bias", nn.initializers.zeros, (self.dim,), self.param_dtype)
        xmeaned = x - x.mean(-1, keepdims=True)
        v = gamma * (xmeaned) / jnp.sqrt((xmeaned**2).mean(-1, keepdims=True)+self.eps)
        v = v + self.lmda * x
        if self.use_bias:
            return v + bias
        return v

class HopfieldTransformer(nn.Module):
    """Full energy transformer"""
    tokdim: int = 768
    nheads: int = 12
    kspace_dim:int = 64
    hidden_ratio:float = 4.
    attn_beta_init: Optional[float] = None
    use_biases_attn:bool = False
    use_biases_chn:bool = False
    param_dtype:Any = jnp.float32

    def setup(self):
        self.attn = MultiheadAttention(
            tokdim=self.tokdim,
            nheads=self.nheads,
            kspace_dim=self.kspace_dim,
            use_bias=self.use_biases_attn,
            beta_init=self.attn_beta_init,
            param_dtype=self.param_dtype
        )
        self.chn = CHNReLU(tokdim=self.tokdim, hidden_ratio=self.hidden_ratio, param_dtype=self.param_dtype)

    def energy(self, g:jnp.ndarray):
        attn_energy = self.attn.energy(g)
        chn_energy = self.chn.energy(g)
        return attn_energy + chn_energy

    def manual_grad(self, g:jnp.ndarray):
        return self.attn.manual_grad(g) + self.chn.manual_grad(g)

    def energy_and_grad(self, g:jnp.ndarray):
        return jax.value_and_grad(self.energy)(g)

#%% Utils
def ifnone(a,b):
    return b if a is None else a

def is_squarable(n:int):
    """Check if 2Darray `x` is square"""
    m = int(np.sqrt(n))
    return m == np.sqrt(n)

def mul(a, b=1):
    return a*b

def normal(key, shape, mean=0, std=1):
    x = jax.random.normal(key, shape)
    return (x * std) + mean

# SUFFIX = '.pickle'
# def save(data, path: Union[str, Path], overwrite: bool = False):
#     """Save a JAX pytree to a file. Saves both static metadata and nodes."""
#     path = Path(path)
#     if path.SUFFIX != SUFFIX:
#         path = path.with_suffix(SUFFIX)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     if path.exists():
#         if overwrite:
#             path.unlink()
#         else:
#             raise RuntimeError(f'File {path} already exists.')
#     with open(path, 'wb') as file:
#         cloudpickle.dump(data, file)


# def load(path: Union[str, Path]):
#     """Load pytree from pickle file"""
#     path = Path(path)
#     if not path.is_file():
#         raise ValueError(f'Not a file: {path}')
#     if path.SUFFIX != SUFFIX:
#         raise ValueError(f'Not a {SUFFIX} file: {path}')
#     with open(path, 'rb') as file:
#         data = cloudpickle.load(file)
#     return data

def load_checkpoint(path: Union[str, Path]):
  load_dict = dict(**np.load(path))
  return load_dict

load_dict = load_checkpoint('./checkpoints/plaindict_ckpt.npz')

class ImageTransformer(eqx.Module):
  patcher: Patcher
  

#%%

class MaskTypeTorch(IntEnum):
    multi_scatter = auto()
    default_scatter = auto()
    perc_25_default = auto()
    perc_25_multi = auto()
    perc_75_default = auto()
    perc_75_multi = auto()
    multi_scatter_patchsize8 = auto()

class Patcher():
    image_shape: Iterable[int]
    patch_size: int
    kh: int
    kw: int

    def __init__(self, image_shape, patch_size, kh, kw):
        """ Class with a `patchify` and `unpatchify` method for a provided image shape.

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
        return rearrange(img, '... c (kh h) (kw w)-> ... (kh kw) c h w', h=self.patch_size, w=self.patch_size)

    def unpatchify(self, patches):
        return rearrange(patches, '... (kh kw) c h w -> ... c (kh h) (kw w)', kh=self.kh, kw=self.kw)

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


#%% 
from dataclasses import dataclass

ckpt_path = "./og_implementation/saved_checkpoints/checkpoint.ckpt"

@dataclass
class Config:
    pretrained_model: str # Path to pretrained model to evaluate
    batch_size: int = 10 # Number of images to evaluate
    img_idx:int = 0
    depth: int = 12
    alpha: float = 0.02
    color_jitter=0.

ckpt = Path("./og_implementation/saved_checkpoints/checkpoint.ckpt") # Behaves like the other pos embeds

conf = Config(ckpt)
#%%
from flax.training import checkpoints
import pickle

checkpoints.restore_checkpoint()
# with open(ckpt, "r") as fp:
#   data = pickle.load(fp)
#%%
def make_dataloaders(self):
    val_batch_size = self.config.batch_size
    if self.config.patch_size == 8:
        assert self.config.mask_type == MaskTypeTorch.multi_scatter_patchsize8, "Ensure you are using the correct masking"
    elif self.config.patch_size == 16:
        pass
    else:
        raise ValueError("We only support patch size 16 and 8 atm")
    patcher = Patcher.from_img_shape(self.config.img_shape, self.config.patch_size)
    # Self.task == classification, use mixup args
    dm = get_default_datamodule(self.config.train_data_dir, self.config.val_data_dir, patcher=patcher, batch_size=self.config.batch_size, masktype=self.config.mask_type, num_workers=self.config.num_workers, auto_augment=self.config.auto_augment, color_jitter=self.config.color_jitter)
    dm.setup()
    train_loader = TorchDataloaderAdapater(dm.train_dataloader)
    val_loader = TorchDataloaderAdapater(dm.val_dataloader)
    return train_loader, val_loader

# Get my dataloaders
#%%
# train_config = TrainConfig("/raid/ILSVRC2012/train/", "/raid/ILSVRC2012/val-by-class/", batch_size=conf.batch_size, depth=conf.depth, alpha=conf.alpha) # Base model has no modifications
# trainer = Trainer(train_config)

# train_dl, val_dl = trainer.make_dataloaders()
# for batch in train_dl():
#     break
# img, mask, c = batch
# mask[mask == 2] = 1
    
# nmask = (mask == 1).sum(-1)[0] # We only want mask == 1
# model = trainer.make_model(nmask)

# lr_fn, opt = trainer.make_optimizer(steps_per_epoch=len(train_dl))
# state = trainer.create_train_state(jax.random.PRNGKey(0), model, opt)
# state = checkpoints.restore_checkpoint(ckpt.parent, state)