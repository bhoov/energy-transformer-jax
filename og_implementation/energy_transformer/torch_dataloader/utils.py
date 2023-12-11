# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04_dataset.utils.ipynb (unless otherwise specified).

__all__ = ['Patcher', 'is_squarable', 'MaskStrategy', 'BlockMask', 'ScatterMask', 'BlockOrScatterMask', 'MaskChooser',
           'MultiScatterMask', 'MultiValueBlockMask', 'MultiScatterOrBlockMask']

import torch
import numpy as np
from einops import rearrange
import functools as ft
from typing import *

def _mul(a,b=1):
    return a*b

def is_squarable(n:int):
    """Check if 2Darray `x` is square"""
    m = int(np.sqrt(n))
    return m == np.sqrt(n)

class MaskStrategy():
    """The only functionality needed to be implemented by a Masking class"""
    def get_mask(self):
        raise NotImplementedError()

class BlockMask(MaskStrategy):
    """A block mask that slides across the image where masked tokens are True.
    If the mask goes over an image boundary, roll over the axis to the other side.

    `nmask` must be a perfect square

    Example:
        import matplotlib.pyplot as plt
        bm = BlockMask(20**2, 32, 32)
        plt.imshow(~bm.get_mask(), cmap="gray")
    """
    def __init__(self, nmask, kh,kw):
        mh = mw = int(np.sqrt(nmask))
        # Ensure nmask is square
        if not is_squarable(nmask):
            print("Warning: Mask is not square, undefined behavior for block masking.")
        assert nmask < (kh * kw), "Mask is bigger than provided image"

        self.kh, self.kw=kh,kw
        self.nmask = nmask
        self.mh, self.mw = mh, mw

    def get_mask(self):
        start_h = torch.randint(self.kh,())
        start_w = torch.randint(self.kw,())

        mask = torch.zeros((self.kh, self.kw), dtype=torch.bool)
        mask[:self.mh, :self.mw] = 1
        mask = torch.roll(mask, (start_h, start_w), (0,1))
        return mask


class ScatterMask(MaskStrategy):
    """A scatter mask where exactly `nmask` are True

    Example:
        import matplotlib.pyplot as plt
        sm = ScatterkMask(20**2, 32, 32)
        plt.imshow(~bm.get_mask(), cmap="gray")
    """
    def __init__(self, nmask, kh, kw):
        self.kh, self.kw = kh, kw
        self.ntok = kh*kw
        self.nmask = nmask

    def get_mask(self):
        # Scatter masking
        a = torch.rand(self.ntok)
        val, idxs = torch.topk(a, self.nmask, sorted=False)
        mask = torch.zeros_like(a, dtype=torch.bool); mask[idxs] = 1
        mask = mask.reshape((self.kh,self.kw))
        return mask

class BlockOrScatterMask(MaskStrategy):
    def __init__(
        self,
        nmask,
        kh,
        kw,
        pscatter=0.5, # Probability to scatter mask vs block mask
    ):
        self.pscatter = pscatter
        self.nmask = nmask

        if pscatter != 1.:
            mh = mw = int(np.sqrt(nmask))
            # Ensure nmask is square
            if mh != np.sqrt(nmask):
                raise ValueError("Mask is not square, undefined behavior for block masking.")
            self.block_masker = BlockMask(nmask, kh, kw)
            self.scatter_masker = ScatterMask(self.block_masker.nmask, kh, kw)
        elif pscatter == 1.:
            self.scatter_masker = ScatterMask(nmask, kh, kw)
        else:
            raise NotImplementedError()

    def get_mask(self):
        if self.pscatter == 1.:
            return self.scatter_masker.get_mask()

        use_scatter = torch.rand(()) < self.pscatter
        mask = (
            self.scatter_masker.get_mask()
            if use_scatter
            else self.block_masker.get_mask()
        )
        return mask

# Cell
class MaskChooser(MaskStrategy):
    def __init__(self, nmask_options, kh, kw, mask_options=[]):
        """Provided a list of {strategy: Strat, prob: Float} options, delegate to appropriate masking each time the block mask is called

        Ensures that the same number of masks is called for each kind of strategy
        """
        self.nmask_options = nmask_options
        self.kh, self.kw = kh, kw
        assert len(mask_options) > 0, "Mask options have not been provided"
        assert sum((m["prob"] for m in mask_options)) == 1., "All probabilities in mask options should sum to 1!"
        sorted_mask_opts = sorted(mask_options, key=lambda strat: -strat["prob"])
        self.mask_options = [(m["prob"], m["strategy"](self.nmask_options, kh, kw)) for m in sorted_mask_opts]

    def get_mask(self):
        pk = torch.rand(())
        curr_p = 0.
        for prob, strat in self.mask_options:
            if pk > 1:
                print(f"BROKEN MASK STRATEGY. RETURNING {strat}")
                return strat.get_mask()
            if pk < (curr_p + prob):
                return strat.get_mask()
            curr_p += prob

# Cell
class MultiScatterMask(MaskStrategy):
    """A nonboolean scatter mask. Typically, used if different kinds of 'patch-filling' are performed by the downstream model"""
    def __init__(
        self,
        nmask_types:Iterable[int],  # Ex: [20,30,2] will create a mask where everything is `0`, 20 items are `1`, 30 items are `2`, and 2 items are `3`
        kh:int, # Number of patches across the height of the image
        kw:int # Number of patches across the width of the image
    ):
        self.kh, self.kw = kh, kw
        self.ntok = kh*kw
        self.nmask_types = nmask_types
        self.tot_mask = sum(self.nmask_types)
        assert 0 < len(self.nmask_types) < 256, "Need at least one value here, length needs to be less than uint8 capacity"
        assert self.tot_mask <= self.ntok, "Cannot mask more items than there are available tokens"

    def get_mask(self) -> torch.tensor:
        with torch.no_grad():
            maskps = torch.rand(self.ntok)
            _,idxs = torch.topk(maskps, self.tot_mask+1, -1)
            mask = torch.zeros_like(maskps, dtype=torch.uint8, requires_grad=False)
            start = 0
            for i,n in enumerate(self.nmask_types):
                stop = start+n
                mask[idxs[start:stop]] = (i+1)
                start=stop
            return mask

class MultiValueBlockMask(MaskStrategy):
    """A nonboolean scatter mask. Typically, used if different kinds of 'patch-filling' are performed by the downstream model"""
    def __init__(
        self,
        nmask_types:Iterable[int],  # Ex: [20,30,2] will create a mask where everything is `0`, 20 items are `1`, 30 items are `2`, and 2 items are `3`
        kh:int, # Number of patches across the height of the image
        kw:int # Number of patches across the width of the image
    ):
        self.kh, self.kw = kh, kw
        self.ntok = kh*kw
        self.nmask_types = nmask_types
        self.tot_mask = sum(self.nmask_types)
        assert 0 < len(self.nmask_types) < 256, "Need at least one value here, length needs to be less than uint8 capacity"
        assert is_squarable(self.tot_mask), "Sum of all provided types must be a perfect square"
        assert self.tot_mask <= self.ntok, "Cannot mask more items than there are available tokens"
        self.mh = self.mw = int(np.sqrt(self.tot_mask))

    def get_mask(self) -> torch.tensor:
        with torch.no_grad():

            start_h = torch.randint(self.kh,())
            start_w = torch.randint(self.kw,())
            mask = torch.zeros((self.kh, self.kw), dtype=torch.int8)
            mask[:self.mh, :self.mw] = 1
            idxs = torch.randperm(self.tot_mask)
            xs,ys = torch.where(mask)

            start = 0
            # The first value has 1 al set, we just need the new values
            for i,n in enumerate(self.nmask_types[1:]):
                stop = start+n
                idxr = idxs[start:stop]
                mask[xs[idxr], ys[idxr]] = (i+2) # Skipping first val
                start = stop

            mask = torch.roll(mask, (start_h, start_w), (0,1))
            return mask

MultiScatterOrBlockMask = ft.partial(MaskChooser, mask_options=[{"strategy":MultiScatterMask, "prob":0.5}, {"strategy":MultiValueBlockMask, "prob":0.5}])