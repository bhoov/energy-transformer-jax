__all__ = ['get_loader', 'get_label_dict', 'imagenet_mean', 'imagenet_std', 'ImagenetNormalizer', 'Patcher',
           'PatchTransform', 'AbstractMaskTransform', 'ScatterMask', 'BlockMask', 'MultiScatterMask',
           'get_default_loader', 'imagenet_unnormalize_image', 'ImagenetDataLoader']

from fastcore import *
from typing import *
import numpy as np
from einops import rearrange
import functools as ft
from .tools import *
from treex import Module

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, NormalizeImage
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, NDArrayDecoder, CenterCropRGBImageDecoder
# from .config import TRAIN_IN1K, VAL_IN1K

# # Helper functions for the notebook
# def get_loader(
#     dstype:str, # 'train' or 'val'
#     pipelines=Dict[str, Any], # Feature processing FFCV pipelines
#     batch_size:int=64,
#     num_workers=40,
#     order: OrderOption=OrderOption.RANDOM, # How shuffling should work
#     ):
#     assert dstype in set(["train", "val"])
#     dsmap = {
#         "train": TRAIN_IN1K,
#         "val": VAL_IN1K,
#     }
#     dspath = dsmap[dstype]
#     loader = Loader(dspath, batch_size=batch_size, num_workers=num_workers,
#                 order=order, pipelines=pipelines)
#     return loader


def get_label_dict():
    """Extract label names corresponding to label indices. Requires a special imagenet_metadata.txt file. Not a crucial function"""
    new_metadata_file = "/raid/ILSVRC2012/imagenet_metadata.txt"
    label_dict = [f.strip().split("\t") for f in open(new_metadata_file, "r").readlines()]
    label_dict = {k:v for k,v in label_dict}
    sorted_keys = sorted(label_dict)
    for i,k in enumerate(sorted_keys):
        label_dict[i] = label_dict[k]
    # Allow indexing by int as well

    return label_dict

imagenet_mean = np.array([0.485, 0.456, 0.406]) * 255
imagenet_std = np.array([0.229, 0.224, 0.225]) * 255
ImagenetNormalizer = ft.partial(NormalizeImage, imagenet_mean, imagenet_std, np.float32)

class Patcher(Module):
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

from ffcv.pipeline.operation import Operation
from typing import *
from abc import abstractmethod
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State
from dataclasses import replace
from ffcv.pipeline.compiler import Compiler

class PatchTransform(Operation):
    """Patch an image in the data loading step"""
    def __init__(self, new_shape:Tuple, patch_size:int):
        super().__init__()
        self.new_shape = new_shape
        self.patch_size = patch_size

    def generate_code(self):
        patch_size = self.patch_size
        def patchify(imgs, dst):
            # Will be faster if I can make this np dependent and set jit_mode=True? Or is einops fast enough?
            dst[:] = rearrange(imgs, 'b (kh h) (kw w) c -> b (kh kw) c h w', h=patch_size, w=patch_size)
            return dst

        return patchify

    def declare_state_and_memory(self, previous_state: State):
        new_state = replace(previous_state, shape=self.new_shape, jit_mode=False)
        mem_allocation = AllocationQuery(self.new_shape, previous_state.dtype)
        return new_state, mem_allocation

    @classmethod
    def from_patcher(cls, patcher:Patcher):
        new_shape = (patcher.n_patches,) + patcher.patch_shape
        patch_size = patcher.patch_size
        return cls(new_shape, patch_size)


# Create a mask transform
class AbstractMaskTransform(Operation):
    """Denotes the loader's MaskTransform. Each sample's mask will be of shape (ntok,) and dtype=np.int8

    Create random masks in the data loading
    """
    def __init__(
        self,
        nmask:Union[int, Sequence[int]], # Right now, assume an int
        kh:int,
        kw:int
    ):
        self.nmask = nmask
        self.kh = kh
        self.kw = kw
        self.ntok = kh*kw
        self.jit_mode = True

    @abstractmethod
    def generate_code(self) -> Callable:
        raise NotImplementedError

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        new_shape = (self.ntok,)
        assert previous_state.shape == (1,), "Must be used right after the decoder"
        new_state = replace(previous_state, shape=new_shape, jit_mode=self.jit_mode)
        mem_allocation = AllocationQuery(new_shape, np.int8)
        return new_state, mem_allocation

class ScatterMask(AbstractMaskTransform):
    """Truly random scatter of masks"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_code(self):
        parallel_range = Compiler.get_iterator()
        nmask, ntok = self.nmask, self.ntok

        def make_mask(x, dst):
            # Ignore `x`, we are initializing the mask right now based on parameters passed to this transform
            bs = x.shape[0]
            dst.fill(np.int8(0))

            for i in parallel_range(bs):
                mask_idxs = np.random.permutation(ntok)[:nmask]
                for j in mask_idxs:
                    dst[i,j] = 1
            return dst

        make_mask.is_parallel = True
        return make_mask


class BlockMask(AbstractMaskTransform):
    """Create a block mask that covers a contiguous square over an image, which rolls over image boundaries to ensure the same number of masked tokens each time"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mh = mw = int(np.sqrt(self.nmask))
        # Ensure nmask is square
        if not is_squarable(nmask):
            raise ValueError("Mask is not square, undefined behavior for block masking.")
        assert nmask < (kh * kw), "Mask is bigger than provided image"
        self.mh, self.mw = mh, mw


    def generate_code(self):
        parallel_range = Compiler.get_iterator()
        nmask, ntok = self.nmask, self.ntok
        kh, kw = self.kh, self.kw
        mh, mw = self.mh, self.mw

        def make_mask(x, dst):
            # Ignore `x`, we are initializing the mask right now based on parameters passed to this transform
            bs = x.shape[0]
            dst.fill(np.int8(0))

            for i in parallel_range(bs):
                start_h, start_w = np.random.randint(kh), np.random.randint(kw)
                for jh in np.arange(start_h, start_h+mh):
                    for jw in np.arange(start_w, start_w + mw):
                        idx = ((kw * jh)%(ntok)) + (jw % kw)
                        dst[i, idx] = 1
            return dst

        make_mask.is_parallel = True
        return make_mask

# class MultiScatterMask(MaskStrategy):
#     """A nonboolean scatter mask. Typically, used if different kinds of 'patch-filling' are performed by the downstream model"""
#     def __init__(
#         self,
#         nmask_types:Iterable[int],  # Ex: [20,30,2] will create a mask where everything is `0`, 20 items are `1`, 30 items are `2`, and 2 items are `3`
#         kh:int, # Number of patches across the height of the image
#         kw:int # Number of patches across the width of the image
#     ):
#         assert 0 < len(self.nmask_types) < 256, "Need at least one value here, length needs to be less than uint8 capacity"
#         assert self.tot_mask <= self.ntok, "Cannot mask more items than there are available tokens"

#     def get_mask(self) -> torch.tensor:
#         with torch.no_grad():
#             maskps = torch.rand(self.ntok)
#             _,idxs = torch.topk(maskps, self.tot_mask+1, -1)
#             mask = torch.zeros_like(maskps, dtype=torch.uint8, requires_grad=False)
#             start = 0
#             for i,n in enumerate(self.nmask_types):
#                 stop = start+n
#                 mask[idxs[start:stop]] = (i+1)
#                 start=stop
#             return mask

class MultiScatterMask(AbstractMaskTransform):
    """Return a mask int32, truly random scatter, but with different values"""
    def __init__(self,
        nmask: Sequence[int], # Ex: [20,30,2] will create a mask where everything is `0`, 20 items are `1`, 30 items are `2`, and 2 items are `3`
        kh:int,
        kw: int
    ):
        super().__init__(nmask, kh, kw)
        assert 0 < len(self.nmask) < 256, "Need at least one value here, length needs to be less than uint8 capacity"
        self.tot_mask = sum(self.nmask)
        assert self.tot_mask <= self.ntok, "Cannot mask more items than there are available tokens"


    def generate_code(self):
        parallel_range = Compiler.get_iterator()
        nmask_types, ntok = np.array(self.nmask, dtype=np.int32), self.ntok

        def make_mask(x, dst):
            # Ignore `x`, we are initializing the mask right now based on parameters passed to this transform
            bs = x.shape[0]
            dst.fill(np.int8(0))
            for i in parallel_range(bs):
                idxs = np.random.permutation(ntok)
                start = 0

                for k in range(len(nmask_types)):
                    nmask = nmask_types[k]
                    stop = start+nmask
                    mask_idxs = idxs[start:stop]
                    for j in mask_idxs:
                        dst[i,j] = k+1
                    start = stop
            return dst

        make_mask.is_parallel = True
        return make_mask



# # Default loader for validation and test
# def get_default_loader(
#     ds_type:str, # 'train' or 'val'
#     *,
#     patcher=None, # If none, use default patcher
#     masktype=None, # Key into optional masking strategy. {"multi_scatter", None, "default_scatter"}
#     **kwargs):
#     if patcher is None:
#         img_size = (3,224,224)
#         patch_size=16
#         patcher = Patcher.from_img_shape(img_size, patch_size)
#     if masktype == "default_scatter" or masktype is None:
#         nmask = 100
#         masker = ScatterMask(nmask, patcher.kh, patcher.kw)
#     elif masktype == "multi_scatter":
#         masker = MultiScatterMask([85,15], patcher.kh, patcher.kw)

#     assert ds_type in set(["train", "val"]), "Please specify a valid dataloader for imagenet"
#     # to_device = JNDToDevice(host_device, non_blocking=non_blocking)
#     default_mask_pipeline = [NDArrayDecoder(), masker]
#     default_label_pipeline = [IntDecoder()]

#     default_train_pipeline = {
#         'image': [RandomResizedCropRGBImageDecoder(patcher.image_shape[1:]), ImagenetNormalizer(), PatchTransform.from_patcher(patcher)],
#         'mask': default_mask_pipeline,
#         'label': default_label_pipeline
#     }

#     default_val_pipeline = {
#         'image': [CenterCropRGBImageDecoder(patcher.image_shape[1:], 0.8), ImagenetNormalizer(), PatchTransform.from_patcher(patcher)],
#         'mask': default_mask_pipeline,
#         'label': default_label_pipeline
#     }
#     if ds_type == "train":
#         return get_loader(ds_type, pipelines=default_train_pipeline, **kwargs)
#     elif ds_type == "val":
#         return get_loader(ds_type, pipelines=default_val_pipeline, **kwargs)

def imagenet_unnormalize_image(a):
    """Unnormalize an image and convert back to 0-255"""
    a = rearrange(a, "... c h w -> ... h w c")
    a = a * imagenet_std + imagenet_mean
    return a.clip(0,255).astype(np.uint8)

from pathlib import Path
class ImagenetDataLoader:
    def __init__(self, root, rel_path_train="train.beton", rel_path_val="val.beton"):
        self.root=Path(root) # Will contain a "train.beton" and a "val.beton"
        self.train_path = self.root / rel_path_train
        self.val_path = self.root / rel_path_val

    def get_loader(self, dstype:str, pipelines, batch_size=64, num_workers=40, order: OrderOption=OrderOption.RANDOM):
        assert dstype in set(["train", "val"])
        dsmap = {
            "train": self.train_path,
            "val": self.val_path
        }
        dspath = dsmap[dstype]
        loader = Loader(dspath, batch_size=batch_size, num_workers=num_workers,
                    order=order, pipelines=pipelines)
        return loader

    def get_default_loader(
        self,
        ds_type:str, # 'train' or 'val'
        *,
        patcher=None, # If none, use default patcher
        masktype=None, # Key into optional masking strategy. {"multi_scatter", None, "default_scatter"}
        **kwargs
    ):
        if patcher is None:
            img_size = (3,224,224)
            patch_size=16
            patcher = Patcher.from_img_shape(img_size, patch_size)
        if masktype == "default_scatter" or masktype is None:
            nmask = 100
            masker = ScatterMask(nmask, patcher.kh, patcher.kw)
        elif masktype == "multi_scatter":
            masker = MultiScatterMask([85,15], patcher.kh, patcher.kw)

        assert ds_type in set(["train", "val"]), "Please specify a valid dataloader for imagenet"
        # to_device = JNDToDevice(host_device, non_blocking=non_blocking)
        default_mask_pipeline = [NDArrayDecoder(), masker]
        default_label_pipeline = [IntDecoder()]

        default_train_pipeline = {
            'image': [RandomResizedCropRGBImageDecoder(patcher.image_shape[1:]), ImagenetNormalizer(), PatchTransform.from_patcher(patcher)],
            'mask': default_mask_pipeline,
            'label': default_label_pipeline
        }

        default_val_pipeline = {
            'image': [CenterCropRGBImageDecoder(patcher.image_shape[1:], 0.8), ImagenetNormalizer(), PatchTransform.from_patcher(patcher)],
            'mask': default_mask_pipeline,
            'label': default_label_pipeline
        }
        if ds_type == "train":
            return self.get_loader(ds_type, pipelines=default_train_pipeline, **kwargs)
        elif ds_type == "val":
            return self.get_loader(ds_type, pipelines=default_val_pipeline, **kwargs)
