# # >>> FOR DEBUGGING ONLY
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4"
# # <<< FOR DEBUGGING ONLY

import os
os.environ["XLA_FLAGS"]="--xla_gpu_force_compilation_parallelism=1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"

import json
import matplotlib.pyplot as plt
from typing import *
import jax
import flax
import flax.linen as nn
import jax.numpy as jnp
import optax
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
from flax import jax_utils, traverse_util
from fastcore.basics import patch
from tqdm import tqdm
import functools as ft
from dataclasses import dataclass, field, replace
from einops import rearrange
import torch
import numpy as np
from torchvision.utils import make_grid
from enum import IntEnum, auto
from simple_parsing.helpers import list_field

from energy_transformer.exp_logger import ExperimentLogger
from energy_transformer.datasets import Patcher, imagenet_unnormalize_image #, get_default_loader
from energy_transformer.flax_modeling import (
    MaskedImageBackbone,
    MaskedImageForClassification,
    MaskedImageForReconstruction,
    DownstreamMaskedImage,
    jax_resize_pos_embed
)
from energy_transformer.torch_dataloader.imagenet import TorchDataloaderAdapater, get_default_datamodule, MaskTypeTorch

from flax.serialization import from_bytes
from flax.core import freeze, unfreeze
from fastcore.meta import delegates
from dataclasses_json import dataclass_json


def make_dummy_input(patcher):
    x = jnp.ones(patcher.patchified_shape())
    mask = jnp.zeros(patcher.n_patches)
    return x, mask

def load_best_pretrained_checkpoint(ckpt_path, img_shape=(3,224,224), patch_size=16, nmask=0):
    patcher = Patcher.from_img_shape(img_shape, patch_size)
    model = MaskedImageForReconstruction.from_backbone_args(patcher, nmask)
    dummy_img, dummy_mask = make_dummy_input(patcher)
    target = model.init(jax.random.PRNGKey(0), dummy_img, dummy_mask)

    with open(ckpt_path, "rb") as fp:
        checkpoint_contents = fp.read()
        params = from_bytes(target, checkpoint_contents)
        
    # Fix position embedding
    new_posemb = target["params"]["backbone"]["pos_embed"]
    old_posemb = params["params"]["backbone"]["pos_embed"]
    print("POS EMBED INFO", old_posemb.shape) 
    print("SHOULD BE NEW shape: ", new_posemb.shape) 
    interp_posemb = jax_resize_pos_embed(old_posemb, new_posemb)
    print("INTERPED IS ", interp_posemb.shape) 
    
    params = unfreeze(params)
    params["params"]["backbone"]["pos_embed"] = interp_posemb
    params = freeze(params)
    return model, params

@delegates(MaskedImageForClassification)
def pretrained_to_cls_head(key:jax.random.PRNGKey, pretrained_model:MaskedImageForReconstruction, pretrained_params, n_classes=1000, nmask=0, **kwargs):
    backbone = replace(pretrained_model.backbone, nmask=nmask)
    model = MaskedImageForClassification(backbone, n_classes, **kwargs)
    dummy_img, dummy_mask = make_dummy_input(model.backbone.patcher)
    new_params = model.init(key, dummy_img)
    new_params = unfreeze(new_params)
    new_params["params"]["backbone"] = pretrained_params["params"]["backbone"]
    new_params = freeze(new_params)
    return model, new_params

@delegates(MaskedImageForClassification)
def cls_task_load_pretrained_checkpoint(
    key:jax.random.PRNGKey,
    ckpt_path,
    n_classes=1000,
    nmask=0,
    img_shape=(3,224,224),
    patch_size=16,
    **kwargs
):
    model, params = load_best_pretrained_checkpoint(ckpt_path, img_shape=img_shape, patch_size=patch_size)
    model, new_params = pretrained_to_cls_head(key, model, params, n_classes=n_classes, nmask=nmask)
    return model, new_params


class TrainTask(IntEnum):
    reconstruction: int = auto()
    classification: int = auto()

class OptimizerType(IntEnum):
    ADAMW: int = auto()
    SGD: int = auto()

class DataLoaderType(IntEnum):
    ffcv: int = auto()
    pytorch: int = auto()

@dataclass_json
@dataclass
class TrainConfig:
    train_data_dir: str # Where to find Training IN1K data
    val_data_dir: str # Where to find Validation IN1K data
    batch_size: int = 768 # Batch size for network
    seed: int = 0 # Random state for initialization
    num_epochs: int = 100 # Number of epochs to run. If restoring from checkpoint, this is adjusted by checkpoint state
    optim: OptimizerType = OptimizerType.ADAMW
    lr_peak_value: float = 5e-4 # Max learning rate
    lr_end_value: float = 5e-7 # Ending learning rate
    lr_init_value: float = 5e-7 # Starting learning rate
    b1: float = 0.9 # For optimizer
    b2: float = 0.99 # For optimizer
    weight_decay: float = 0.05 # For optimizer
    n_warmup_epochs: int = 2 # Number of warmup epochs
    task: TrainTask = TrainTask.reconstruction # What the task is (currently only reconstruction is supported)
    mask_type: MaskTypeTorch = MaskTypeTorch.multi_scatter # masking type
    num_workers: int = 100 # Number of workers to use for dataloading
    log_every_steps: Optional[int] = 50 # When to check loss and checkpoint saving
    img_reconstructions_every: int = 800  # Reconstruct the image every N training batches
    img_shape: Tuple[int, int, int] = list_field(3, 224, 224) # Shape of the image
    patch_size: int = 16 # Size of each patch 
    force_git_unclean: bool = True # If True, the code will not issue a warning if the git repo is unclean
    devices: List[int] = list_field() # List of devices to use for training. If none, use all devices
    dl_type: DataLoaderType = DataLoaderType.pytorch # The type of dataloader to use
    n_classes: int = 1000 # Number of classes in classification task
    head_type: str = "mlp" # Type of head. Either "linear" or "mlp". Only used for classification task
    cls_from_ckpt: Optional[str] = None # if the task is "classification" and this is provided, load this checkpoint for finetuning
    no_autoresume: bool = False # If given, do NOT try to autoreload from the provided checkpoint directory
    grad_global_clip: float = 1. # Clip gradients at this value. Ignore if this value == 0
    scale_lr_by_batchsize: bool = False # Scale the peak learning rate by batchsize (batchsize/256)*peak_lr
    tokdim: int = 768 # Token dimension of the base model                                                                                                                     
    nheads: int = 12 # Number of heads in the base model
    kspace_dim:int = 64 # Size of the attention operation dimensino
    hidden_ratio:float = 4. # Size of the hidden layer in the CHN
    attn_beta_init: Optional[float]=None # Default `beta` in the attention
    use_biases_attn:bool = False # Do we use biases in the energy attention?
    use_biases_chn:bool = False # Do we use biases in the CHN operation?
    use_biases_norm:bool = True # Do we allow our layer norm to use biases?
    eps:float = 1e-05 # Floating point error
    alpha: float = 0.1 # Step size of the gradient descent
    depth: int = 12 # How deep our block is run
    
    color_jitter: float = 0.4 # Part of transform
    auto_augment: Optional[str] = None # None or 'rand' for RandAug

    smoothing: float = 0.1 # For classification, how much smoothing to apply to the labels
    
    def to_file(self, fname):
        with open(fname, "w") as fp:
            json.dump(self.to_dict(), fp)
            
    @classmethod
    def from_file(cls, fname):
        return cls.from_json(open(fname,'r').read())
        

## Training Helpers
@patch
def patchified_shape(p: Patcher):
    """Return the expected size output from the data generator"""
    return (p.n_patches, *p.patch_shape)


def ifnone(a, b):
    if a is None:
        return b
    return a


def cross_entropy_loss(*, logits, labels, num_classes=1000, alpha=0.1):
    labels_onehot = jax.nn.one_hot(labels, num_classes=num_classes)
    labels_smooth = optax.smooth_labels(labels_onehot, alpha)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_smooth).mean()


def mse_patch_loss(preds, truth, mask):
    # preds in BNCHW
    # truth in BNCHW
    occluded_patches = (mask > 0)[:, :, None, None, None]
    yguess = preds * occluded_patches
    ytruth = truth * occluded_patches
    loss = jnp.sqrt(jnp.mean((yguess - ytruth) ** 2))
    return loss

def distribute_data(xs, n_devices=None):
    """Distribute input batch across n_devices input batch from tf Tensors to numpy arrays."""
    n_devices = ifnone(n_devices, jax.device_count())

    def _prepare(x):
        return rearrange(x, "(device batch) ... -> device batch ...", device=n_devices)

    return jax.tree_map(_prepare, xs)

def compute_metrics_classification(*, logits, labels, alpha=0.1):
    loss = cross_entropy_loss(logits=logits, labels=labels, alpha=alpha)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    metrics = jax.lax.pmean(metrics, axis_name="device")
    return metrics

def compute_metrics_reconstruction(*, preds, truth, mask):
    loss = mse_patch_loss(preds=preds, truth=truth, mask=mask)
    metrics = {
        "loss": loss,
    }
    metrics = jax.lax.pmean(metrics, axis_name="device")
    return metrics

class FFCVDataloaderAdapter:
    def __init__(self, dl):
        self.dl = dl
        self.N = len(self.dl)

    def __len__(self):
        return self.N

    def __call__(self):
        return self.dl

class Trainer:
    def __init__(self, config:TrainConfig):
        self.config = config
        
        # For convenience
        self.task = self.config.task

        # Checking
        assert (self.config.batch_size % self.n_devices) == 0, f"Batch size must be divisible by the number of devices {self.n_devices}"


    @ft.cached_property
    def devices(self):
        if len(self.config.devices) == 0:
            # Use all devices if no devices are specified
            devices = list(range(jax.device_count()))
        else:
            devices = self.config.devices
        return devices

    @ft.cached_property
    def n_devices(self):
        return len(self.devices)

    def make_optimizer(
        self,
        steps_per_epoch: int,
    ):
        total_steps = self.config.num_epochs * steps_per_epoch
        peak_value = self.config.lr_peak_value if not self.config.scale_lr_by_batchsize else self.config.lr_peak_value * (self.config.batch_size / 256)
        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=self.config.lr_init_value,
            peak_value=peak_value,
            warmup_steps=int(self.config.n_warmup_epochs * steps_per_epoch),
            decay_steps=total_steps,
            end_value=self.config.lr_end_value,
        )

        if self.config.optim == OptimizerType.ADAMW:
            optimizer = optax.adamw(scheduler, b1=self.config.b1, b2=self.config.b2, weight_decay=self.config.weight_decay)
        elif self.config.optim == OptimizerType.SGD:
            raise NotImplementedError(f"Optimizer type {self.config.optim} should be supported but is not yet.")

        if self.config.grad_global_clip != 0.:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.grad_global_clip),
                optimizer
            )

        return scheduler, optimizer


    def make_dataloaders(self):
        val_batch_size = self.config.batch_size

        if self.config.dl_type == DataLoaderType.ffcv:
            raise ValueError("We no longer support FFCV dataloading")
            train_loader = get_default_loader(
                "train", batch_size=self.config.batch_size, masktype=self.config.mask_type, num_workers=self.config.num_workers
            )
            val_loader = get_default_loader(
                "val", batch_size=val_batch_size, masktype=self.config.mask_type, num_workers=self.config.num_workers
            )
            return FFCVDataloaderAdapter(train_loader), FFCVDataloaderAdapter(val_loader)
        elif self.config.dl_type == DataLoaderType.pytorch:
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

    def init_step(self, key, model: DownstreamMaskedImage):
        """Initialize parameters of the model"""
        img_shape = model.backbone.patcher.patchified_shape()
        mask_shape = (model.backbone.patcher.n_patches,)

        @jax.jit
        def init(*args):
            return model.init(*args)

        img = jnp.ones(img_shape, model.backbone.dtype)
        mask = jnp.zeros(mask_shape, model.backbone.dtype)
        if self.task == TrainTask.classification:
            variables = init({"params": key}, img)
        elif self.task == TrainTask.reconstruction:
            variables = init({"params": key}, img, mask)
        return variables["params"]

    def eval_step(self, state, batch):
        img, mask, label = batch
        variables = {"params": state.params}

        if self.task == TrainTask.reconstruction:
            preds = state.apply_fn(variables, img, mask)
            metrics = compute_metrics_reconstruction(preds=preds, truth=img, mask=mask)
        elif self.task == TrainTask.classification:
            preds = state.apply_fn(variables, img)
            metrics = compute_metrics_classification(logits=preds, labels=label[:,0], alpha=self.config.smoothing)

        return metrics

    def train_step(self, state, batch, learning_rate_fn):
        """Perform a single training step."""
        img, mask, label = batch

        def loss_fn(params):
            """loss function used for training."""

            if self.task == TrainTask.classification:
                preds = state.apply_fn({"params": params}, img)
                loss = cross_entropy_loss(logits=preds, labels=label[:,0], alpha=self.config.smoothing)
            elif self.task == TrainTask.reconstruction:
                preds = state.apply_fn({"params": params}, img, mask)
                loss = mse_patch_loss(preds, img, mask)

            return loss, preds

        step = state.step
        lr = learning_rate_fn(step)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, preds), grads = grad_fn(state.params)

        grads = jax.lax.pmean(grads, axis_name="device")

        if self.task == TrainTask.reconstruction:
            metrics = compute_metrics_reconstruction(preds=preds, truth=img, mask=mask)
        elif self.task == TrainTask.classification:
            metrics = compute_metrics_classification(logits=preds, labels=label[:,0], alpha=self.config.smoothing)

        metrics["learning_rate"] = lr
        new_state = state.apply_gradients(grads=grads)
        return new_state, metrics   

    def create_train_state(
        self,
        rng,
        model: DownstreamMaskedImage,
        optimizer,
    ):
        """Create initial training state."""
        params = self.init_step(rng, model)
        if self.task == TrainTask.reconstruction:
            apply_fn = jax.vmap(model.apply, in_axes=(None, 0, 0))
        elif self.task == TrainTask.classification:
            apply_fn = jax.vmap(model.apply, in_axes=(None, 0))
        state = TrainState.create(apply_fn=apply_fn, params=params, tx=optimizer)
        return state

    def make_model(self, nmask:int):
        patcher = Patcher.from_img_shape(self.config.img_shape, self.config.patch_size)
        print("Patcher: ", patcher.n_patches, patcher.kh, patcher.kw)
        backbone_kwargs = dict(
            tokdim=self.config.tokdim,
            nheads=self.config.nheads,
            kspace_dim=self.config.kspace_dim,
            hidden_ratio=self.config.hidden_ratio,
            attn_beta_init=self.config.attn_beta_init,
            use_biases_attn=self.config.use_biases_attn,
            use_biases_chn=self.config.use_biases_chn,
            use_biases_norm=self.config.use_biases_norm,
            eps=self.config.eps,
            alpha=self.config.alpha,
            depth=self.config.depth
        )
        if self.config.task == TrainTask.reconstruction:
            model = MaskedImageForReconstruction.from_backbone_args(
                patcher, nmask, **backbone_kwargs
            )
        elif self.config.task == TrainTask.classification:
            model = MaskedImageForClassification.from_backbone_args(patcher, n_classes=self.config.n_classes, head_type=self.config.head_type, **backbone_kwargs)
        return model


class TrainState(train_state.TrainState):
    # task: TrainTask = TrainTask.reconstruction
    pass


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def make_local(obj):
    """For distributed pytrees, return the pytree living on the local host"""
    return jax.device_get(jax.tree_map(lambda x: x[0], obj))


def save_checkpoint(state, workdir):
    if jax.process_index() == 0:
        # get train state from the first replica
        state = make_local(state)
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=3)


## Metric calculations
def show_predictions(patcher, imgs, nv=16, nrow=None):
    """Turn patchified predictions into image grid"""
    imgs = [
        torch.tensor(np.array(imagenet_unnormalize_image(patcher.unpatchify(x[:nv]))))
        for x in imgs  # HWC images
    ]
    if nrow is None:
        nrow = int(np.ceil(np.sqrt(len(imgs[0]))))

    plot_img = make_grid(
        rearrange(imgs, "s b h w c -> b c h (s w)"),
        nrow=nrow,
        normalize=False,
    )
    return plot_img


def img_results_for_logging(patcher: Patcher, batch, yhat, nv=16, nrow=None):
    """Usage:
    model = ...
    batch = ... #(img, mask, label)
    yhat = ... # prediction from img and mask

    writer = SummaryWriter("text_exp5")
    predgrid = img_results_for_logging(model.patcher, batch, yhat)

    writer.add_image("Preds", predgrid, 0)
    """
    img, mask, label = batch
    img, mask = img[:nv], mask[:nv]
    yhat = yhat[:nv]
    pred_grid = _img_results_for_logging(patcher, img, mask, yhat, nv=nv, nrow=nrow)
    return pred_grid


def _img_results_for_logging(patcher: Patcher, img, mask, yhat, nv=16, nrow=None):
    img, mask, yhat = img[:nv], mask[:nv], yhat[:nv]
    masked_img_tokens = jnp.einsum(
        "bnchw,bn->bnchw", img, mask != 1
    )  # Show input to model
    reconstructed_patches_only = jnp.einsum(
        "bnchw,bn->bnchw", yhat, mask > 0
    )  # Only show decodings
    given_patches = mask == 0
    new_info = yhat.at[given_patches].set(img[given_patches])
    pred_grid = show_predictions(
        patcher,
        [masked_img_tokens, yhat, reconstructed_patches_only, new_info, img],
        nv=nv,
        nrow=nrow,
    )
    return pred_grid

def gridify_img_results(patcher: Patcher, imgs: List[Any], nv=16, nrow=None):
    imgs = [x[:nv] for x in imgs]
    # masked_img_tokens = jnp.einsum(
    #     "bnchw,bn->bnchw", img, mask != 1
    # )  # Show input to model
    # reconstructed_patches_only = jnp.einsum(
    #     "bnchw,bn->bnchw", yhat, mask > 0
    # )  # Only show decodings
    # given_patches = mask == 0
    # new_info = yhat.at[given_patches].set(img[given_patches])
    pred_grid = show_predictions(
        patcher,
        imgs,
        nv=nv,
        nrow=nrow,
    )
    return pred_grid

def get_devices(idxs:List[int]):
    dev = jax.devices()
    return [dev[i] for i in idxs]

def train_and_evaluate(workdir: str, config: TrainConfig) -> TrainState:
    """Execute model training and evaluation loop.
    Args:
        config: Hyperparameter configuration for training and evaluation.
        workdir: Directory where the tensorboard summaries are written to.

    Returns:
        Final TrainState.
    """
    # print(f"Starting JAX multi-node service with config: {config.mp}")
    # print(f'process {config.mp.host_idx} starting.')
    # service = config.mp.connect_to_gpu_cluster()
    # print(f"I am {config.mp.server_ip}. I see {jax.local_device_count()}/{jax.device_count()} devices")
    print(f"{jax.device_count()} available devices: {jax.devices()}")
    print(f"Each node has {jax.local_device_count()} available devices")

    print(f"Training model in `{workdir}` with config: ", config)
    trainer = Trainer(config)

    train_loader, val_loader = trainer.make_dataloaders()
    steps_per_epoch = len(train_loader)
    learning_rate_fn, opt = trainer.make_optimizer(steps_per_epoch=steps_per_epoch)

    # Create model and state
    rng = jax.random.PRNGKey(config.seed)
    if config.task == TrainTask.reconstruction:
        print("Creating reconstruction model...")
        # Create dummy inputs
        for batch in val_loader():
            img, mask, label = batch
            break
        nmask = (mask > 0).sum(-1)[0]
        model = trainer.make_model(nmask)
    elif config.task == TrainTask.classification:
        if config.cls_from_ckpt is not None:
            from_ckpt = config.cls_from_ckpt
            print(f"Loading classification model from `{from_ckpt}`")
            model, params = cls_task_load_pretrained_checkpoint(rng, from_ckpt, n_classes=config.n_classes, img_shape=config.img_shape, patch_size=config.patch_size, head_type=config.head_type)
            params = params["params"] # It loads with this as root key, init does not return this
        else:
            print(f"Creating classification model from scratch")
            model = trainer.make_model(0)
    state = trainer.create_train_state(rng, model, opt)

        
    @jax.jit
    def predict(params, batch):
        img, mask, label = batch
        return state.apply_fn({"params": params}, img, mask)

    logger = ExperimentLogger(workdir, keep_best=3).init(use_time_dir=False, force_yes=config.force_git_unclean)
    logger.save_hyperparams(config.to_dict())
    if config.no_autoresume:
        print("No autoresume requested. Checking to make sure this is a clean logging directory")
        assert len(list(logger.checkpoint_dir.glob("*"))) == 1, f"{logger.checkpoint_dir} should have no saved checkpoints. Please provide a clean logging directory"
    else:
        print(f"Trying to reload state from {logger.checkpoint_dir}. Will only load if a checkpoint is provided.")
        state = restore_checkpoint(state, logger.checkpoint_dir)
    state = jax_utils.replicate(state, get_devices(jnp.array(trainer.devices)))

    @ft.partial(jax.pmap, axis_name="device")
    def p_train_step(state, batch):
        print("JITTING train step...")
        output = trainer.train_step(state, batch, learning_rate_fn=learning_rate_fn)
        return output

    @ft.partial(jax.pmap, axis_name="device")
    def p_eval_step(state, batch):
        print("JITTING eval step...")
        output = trainer.eval_step(state, batch)
        return output 

    train_metrics = []
    try:
        print(f"Current state has step: ", state.step[0])
        state_step = int(state.step)
    except TypeError:
        # The state is distributed
        state_step = int(make_local(state.step))
    start_epoch = state_step // steps_per_epoch
    print(
        f"Starting training at epoch {start_epoch}, going to epoch {config.num_epochs}"
    )
    for epoch in tqdm(range(start_epoch, config.num_epochs)):
        state_step = int(make_local(state).step)
        logger.writer.add_scalar("epoch", epoch, state_step)
        train_pbar = tqdm(
            train_loader(),
            desc="training",
            unit="batch",
            leave=False,
        )

        for step, batch in enumerate(train_pbar):
            state_step = int(make_local(state).step)
            dev_batch = distribute_data(batch, trainer.n_devices)
            state, metrics = p_train_step(state, dev_batch)

            if config.log_every_steps is not None:
                train_metrics.append(metrics)  # list of {loss, learning_rate}
                if (step + 1) % config.log_every_steps == 0:
                    ploss = jnp.mean(jnp.stack([m["loss"] for m in train_metrics]))
                    logger.writer.add_scalar("train.loss", ploss, state_step)
                    description = f"Epoch={epoch}, Step={step}, loss={ploss}"
                    logger.writer.add_scalar("train.lr", make_local(metrics["learning_rate"]).item(), state_step)

                    if config.task == TrainTask.classification:
                        pacc = jnp.mean(jnp.stack([m["accuracy"] for m in train_metrics]))
                        description += f", acc={pacc}"
                        logger.writer.add_scalar("train.acc", pacc, state_step)

                    train_pbar.set_description("Training: ", description)
                    train_metrics = [] # Reset Metrics

                    if config.task == TrainTask.reconstruction:
                        logger.flax_save_checkpoint(make_local(state), state_step, -ploss) # Check loss against current loss state
                    elif config.task == TrainTask.classification:
                        logger.flax_save_checkpoint(make_local(state), state_step, pacc) # Main metric is the accuracy
 
                    desc = f"SAVED state checkpoint at: " + description
                    train_pbar.set_description(desc)

            if config.task == TrainTask.reconstruction and (step + 1) % config.img_reconstructions_every == 0:
                # Save img reconstruction
                batch0 = make_local(dev_batch)
                params0 = make_local(state.params)
                yhat = predict(params0, batch0)
                predgrid = img_results_for_logging(model.backbone.patcher, batch0, yhat)
                logger.writer.add_image("train.reconstructions", predgrid, state_step)

        val_losses = []
        val_accs = []
        for test_step, batch in enumerate(
            tqdm(
                val_loader(),
                desc="validating",
                unit="batch",
                leave=False,
            )
        ):
            state_step = int(make_local(state).step)
            dev_batch = distribute_data(batch, trainer.n_devices)
            eval_metrics = p_eval_step(state, dev_batch)
            val_losses.append(jnp.mean(eval_metrics["loss"]))
            if config.task == TrainTask.classification:
                val_accs.append(jnp.mean(eval_metrics["accuracy"]))

            if config.task == TrainTask.reconstruction and test_step == 0:
                batch0 = make_local(dev_batch)
                param0 = make_local(state.params)
                yhat = predict(param0, batch0)
                predgrid = img_results_for_logging(model.backbone.patcher, batch0, yhat)
                logger.writer.add_image("val.reconstructions", predgrid, state_step)

        vloss = jnp.mean(jnp.stack(val_losses))
        logger.writer.add_scalar("val.loss", vloss, state_step)

        if config.task == TrainTask.classification:
            vacc = jnp.mean(jnp.stack(val_accs))
            logger.writer.add_scalar("val.acc", vacc, state_step)

    return state
