from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tfms

from torchvision.datasets.folder import default_loader
import pytorch_lightning as pl
import numpy as np
import random
from .utils import ScatterMask, BlockMask, MaskStrategy, MultiScatterMask

from pathlib import Path
import torch
import numpy as np
import functools as ft
from typing import *

from energy_transformer.datasets import Patcher
from energy_transformer.config import TRAIN_IN1K_torch, VAL_IN1K_torch

from enum import IntEnum, auto
from timm.data.loader import create_transform
from fastcore.meta import delegates


def ifnone(a, b):
    if a is None: return b
    return a

## Helpers for transforming and loading
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

normalizer = tfms.Normalize(
    mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD)
)

transform_opts = dict(
    train=create_transform(224, is_training=True),
    val=create_transform(224, is_training=False)
)


def show_tensor(x):
    n = len(x.shape)
    im_mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=x.device)
    im_std = torch.tensor(IMAGENET_DEFAULT_STD, device=x.device)
    if n == 4:
        im_mean = im_mean[None]
        im_std = im_std[None]
    for _ in range(2):
        im_mean = im_mean.unsqueeze(-1)
        im_std = im_std.unsqueeze(-1)
    x = x * im_std + im_mean
    return x


class PatchAndMaskDatasetOutput(TypedDict):
    img_tokens: torch.tensor
    mask: torch.tensor
    masked_img_tokens: torch.tensor
    target:int



def check_mean_and_std(mean, std):
    if mean is None:
        assert std is None, "Must provide both mean and std if using at all"
    elif std is None:
        assert mean is None, "Must provide both mean and std if using at all"



class PatchAndMaskImageNetDataset(ImageFolder):
    """Custom data for autoencoding, with dataloading taking care of the masking.

    Handles the initial patching and the masking

    Usage:
        ```
        patcher = Patcher.from_img_shape((3,224,224), 4)
        ds = PatchAndMaskImageNetDataset(ILSVRC / "Data" / "CLS-LOC" / "train", patcher=patcher)
        ```
    """

    def __init__(
        self,
        root: str,
        patcher: Patcher,
        masker: MaskStrategy,
        transform=None,
        limit_to=None,
        shuffle_samples=False,
    ):
        """
        Args:
            root: Root directory containing images in the kaggle structure
            patcher: Patchifier that turns the image into tokens
            pmask: Number of tokens that are masked
            pscatter: How often to mask in random scatter. For anything that is not masked in a random scatter, use block masking
            dl_seed: Seed the shuffling of the training and validation datasets to ensure the same results each time
            shuffle_samples: Shuffle the samples. Recommended if you are testing on a subset of the validation dataset
        """
        super().__init__(root)

        self.image_shape = patcher.image_shape
        self.limit_to = limit_to

        self.patcher = patcher
        self.masker = masker

        self.transform = ifnone(transform, tfms.ToTensor())
        self.shuffle_samples=shuffle_samples

        if shuffle_samples:
            random.shuffle(self.samples)

    def __len__(self):
        if self.limit_to is not None:
            return min(self.limit_to, len(self.samples))
        return len(self.samples)

    def __getitem__(self, idx:int) -> PatchAndMaskDatasetOutput:
        path, target = self.samples[
            idx
        ]  # target is unused since we are not concerned with labels
        sample = self.loader(path)
        sample = self.transform(sample)

        # Now patchify
        mask = self.masker.get_mask()
        img_tokens = self.patcher.patchify(sample)
        with torch.no_grad():
            mask_tokens = mask.flatten()
            masked_img_tokens = img_tokens.clone()
            masked_img_tokens[mask_tokens > 0] = 0.0

        return dict(
            img_tokens=img_tokens.to(torch.float32),
            masked_img_tokens=masked_img_tokens.to(torch.float32),
            mask=mask_tokens,
            target=target,
        )

    def to_img(self, tokens):
        return self.patcher.unpatchify(tokens)


PatchAndMaskImageNetTrain = ft.partial(
    PatchAndMaskImageNetDataset,
    transform=transform_opts["train"],
)
PatchAndMaskImageNetValAndTest = ft.partial(
    PatchAndMaskImageNetDataset, transform=transform_opts["val"]
)

class PatchAndMaskImageNetDataLoaders():
    def __init__(
        self,
        train_path: Union[Path, str],
        val_path: Union[Path, str],
        patcher: Patcher,
        masker: MaskStrategy, 
        masker_val=None,
        batch_size=32,
        n_val=None,
        num_workers=16,
        dl_seed=42,
        train_tfms=None,
        val_tfms=None
    ):
        self.batch_size = batch_size
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.limit_val_to = n_val
        self.num_workers = num_workers
        self.patcher = patcher
        self.masker = masker
        self.masker_val = ifnone(masker_val, masker)

        self.dl_seed = dl_seed
        self.train_tfms= ifnone(train_tfms, transform_opts["train"])
        self.val_tfms= ifnone(val_tfms, transform_opts["val"])
        
        self.ds_train = PatchAndMaskImageNetDataset(
            self.train_path,
            patcher=self.patcher,
            masker=self.masker,
            transform=self.train_tfms,
        )
        self.seed(self.dl_seed)
        self.ds_val = PatchAndMaskImageNetDataset(
            self.val_path,
            patcher=self.patcher,
            masker=self.masker,
            limit_to=self.limit_val_to,
            shuffle_samples=self.limit_val_to is not None,
            transform=self.val_tfms,
        )
                
    @staticmethod
    def seed(seed):
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


class PatchAndMaskImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: Union[Path, str],
        val_path: Union[Path, str],
        patcher: Patcher,
        masker: MaskStrategy, 
        masker_val=None,
        batch_size=32,
        n_val=None,
        num_workers=16,
        dl_seed=42,
        train_tfms=None,
        val_tfms=None,
        shuffle_val=False
    ):
        """
        Args:
            train_path: Path to folder containing training img folders of training imgs
            val_path: Path to folder containing val img folders of val imgs
            patcher: The patcher that is used to split images into tokens
            masker:

        Usage:
            ```
            dm = PatchAndMaskTinyImageNetDataModule("../data/ImageNet/", patcher, num_workers=32)
            dm.setup()
            for batch in dm.train_dataloader():
            ```
                ...
        """
        super().__init__()
        self.batch_size = batch_size
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.limit_val_to = n_val
        self.num_workers = num_workers
        self.patcher = patcher
        self.masker = masker
        self.masker_val = ifnone(masker_val, masker)

        self.dl_seed = dl_seed
        self.train_tfms= ifnone(train_tfms, transform_opts["train"])
        self.val_tfms= ifnone(val_tfms, transform_opts["val"])
        self.shuffle_val = shuffle_val
        
    @staticmethod
    def seed(seed):
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def prepare_data(self):
        # We will not download this algorithmically, but instead include it as part of the setup to our environment
        pass

    def setup(self, stage=None):
        self.ds_train = PatchAndMaskImageNetDataset(
            self.train_path,
            patcher=self.patcher,
            masker=self.masker,
            transform=self.train_tfms,
        )
        self.seed(self.dl_seed)
        self.ds_val = PatchAndMaskImageNetDataset(
            self.val_path,
            patcher=self.patcher,
            masker=self.masker,
            limit_to=self.limit_val_to,
            shuffle_samples=self.limit_val_to is not None,
            transform=self.val_tfms,
        )

    def train_dataloader(self):
        # self.seed(self.dl_seed+1)
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=self.shuffle_val,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True
        )

class MaskTypeTorch(IntEnum):
    multi_scatter = auto()
    default_scatter = auto()
    perc_25_default = auto()
    perc_25_multi = auto()
    perc_75_default = auto()
    perc_75_multi = auto()
    multi_scatter_patchsize8 = auto()

@delegates(create_transform)
def get_default_datamodule(
    train_dir, val_dir,
    img_shape:int=224,
    patch_size=16,
    patcher=None, # If none, use default patcher. Overrides img_shape and patch_size
    masktype:MaskTypeTorch=MaskTypeTorch.default_scatter, # Key into optional masking strategy. {"multi_scatter", None, "default_scatter"}
    batch_size=32,
    num_workers=32,
    shuffle_val=False,
    **kwargs
    ):
    if patcher is None:
        if isinstance(img_shape, int):
            img_size = (3, img_shape, img_shape)
        elif isinstance(img_shape, tuple) and len(img_shape) == 2:
            img_size = (3, *img_shape)
        else:
            raise ValueError(f"Got `{img_shape}` for img_shape, expected into or len 2 tuple")

        patcher = Patcher.from_img_shape(img_size, patch_size)

    if masktype == MaskTypeTorch.default_scatter:
        print("DEFAULT SCATTERING")
        nmask = 100
        masker = ScatterMask(nmask, patcher.kh, patcher.kw)
    elif masktype == MaskTypeTorch.multi_scatter:
        print("MULTISCATTERING")
        masker = MultiScatterMask([90,10], patcher.kh, patcher.kw)
    elif masktype == MaskTypeTorch.perc_25_default:
        print("DEFAULT 25")
        nmask = 50 
        masker = ScatterMask(nmask, patcher.kh, patcher.kw)
    elif masktype == MaskTypeTorch.perc_75_default:
        print("DEFAULT 75")
        nmask = 150 
        masker = ScatterMask(nmask, patcher.kh, patcher.kw)
    elif masktype == MaskTypeTorch.perc_25_multi:
        print("MULTI 25 MASK")
        masker = MultiScatterMask([45,5], patcher.kh, patcher.kw)
    elif masktype == MaskTypeTorch.perc_75_multi:
        print("MULTI 75 MASK")
        masker = MultiScatterMask([135,15], patcher.kh, patcher.kw)
    elif masktype == MaskTypeTorch.multi_scatter_patchsize8:
        print("PATCHSIZE 8 masking")
        masker = MultiScatterMask([360,40], patcher.kh, patcher.kw)
    else:
        raise ValueError(f"Mask type {masktype} is not supported")
        
    train_tfms = create_transform(patcher.image_shape[-1], is_training=True, **kwargs)
    val_tfms = create_transform(patcher.image_shape[-1], **kwargs)

    dm = PatchAndMaskImageNetDataModule(train_dir, val_dir, patcher,masker, batch_size=batch_size, num_workers=num_workers, train_tfms=train_tfms, val_tfms=val_tfms, shuffle_val=shuffle_val) 
    dm.setup()
    return dm


def torchbatch_to_numpy(batch):
    img = batch["img_tokens"]
    mask = batch["mask"]
    label = batch["target"]
    
    # Transform into numpy arrays
    img = img.numpy()
    mask = np.array(mask, dtype=np.int8)
    label = np.array(label, dtype=np.int32)[..., None]
    return img, mask, label

class TorchDataloaderAdapater:
    def __init__(self, dl_gen):
        self.dl_gen = dl_gen
        self.N = len(dl_gen())

    def __len__(self):
        return self.N
    
    def __call__(self):
        for batch in self.dl_gen():
            yield torchbatch_to_numpy(batch)
