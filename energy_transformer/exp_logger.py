__all__ = ['repo', 'GitNotReadyError', 'ExperimentLogger']

import pygit2 as git
from pathlib import Path
import os
from fastcore.basics import patch
from typing import *
from flax.training import checkpoints

@patch
def ignored_status(r:git.Repository):
    """Fetch the same information as `git status` would give you given the gitignore"""
    states = repo.status()
    return {k:v for k,v in states.items() if not r.path_is_ignored(k)}

@patch
def has_changes(r:git.Repository):
    return len(r.ignored_status()) > 0

@patch
def current_commit(r:git.Repository):
    return r.head.target

try:
    # in `enfami/exp_logger.py`
    ROOT = Path(os.path.abspath(__file__)).parent.parent
except NameError:
    # __file__ is not defined. In jupyter notebook in `nbs` dir
    ROOT = Path(os.path.abspath(".")).parent
    print("In a jupyter notebook. Root dir: ", ROOT)

try:
    repo = git.Repository(ROOT)
except:
    repo = None

from datetime import datetime
from dataclasses import dataclass, replace
import shutil
import json
from tensorboardX import SummaryWriter
from .tools import save, load
from flax.training import checkpoints
from flax.training import train_state
from flax.errors import InvalidCheckpointError

class GitNotReadyError(Exception): pass

@dataclass
class ExperimentLogger():
    """Module to handle logging experiments.

    While not enforced by the dataclass, treat this object as immutable.
    If the logger is ever modified, it should be returned as a new object

    Example:
    ```
    logger = ExperimentLogger("mylogs")
    exp = logger.init("first-try", "Trying to see why my loss always becomes NaN")
    ```
    """
    base_dir: Path
    git_repo: git.Repository = repo # Git repository containing this code
    exp_dir: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None
    hyperparam_dir: Optional[Path] = None
    best_checkpoint_dir: Optional[Path] = None
    writer: Optional[SummaryWriter] = None
    keep_best: int = 1 # How many checkpoints to keep
    metric_calculator: Callable[[Any,Any],bool] = lambda metric: metric # Convert a metric into a scalar that can be sorted. Lower values are considered "better"

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = {}
        self._checkpoint_stack = []
        self._ckpt_uid = 0
        return self

    def init(
        self,
        use_time_dir=True,
        exp_name:Optional[str]=None, # Name of the experiment. Will be appended to current date and time
        description=None, # Short description of experiment
        force_yes=True, # If True, do not prompt user for confirmation about git
    ):
        """Create a new experiment directory"""

        dirty_repo = False
        if self.git_repo is not None:
            if self.git_repo.has_changes():
                dirty_repo = True
                if not force_yes:
                    user_inp = input("It looks like your current git repository has unstaged changes. Are you sure you want to continue? [y/N]")
                    if "y" not in user_inp.lower():
                        raise GitNotReadyError(f"Git was not clean. Please clean up first: {self.git_repo.ignored_status()}")

        curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        if use_time_dir:
            exp_dir_name = curr_time if exp_name is None else f"{curr_time}_{exp_name}"
            exp_dir = Path(self.base_dir / exp_dir_name)
        else:
            exp_dir = Path(self.base_dir)

        new = replace(self, exp_dir=exp_dir)
        new = replace(new, writer=SummaryWriter(new.exp_dir)) # Will create the experiment dir if it hasn't yet been created
        new = replace(new, checkpoint_dir=(new.exp_dir / "checkpoints"))
        new.checkpoint_dir.mkdir(exist_ok=True)
        new = replace(new, best_checkpoint_dir=(new.checkpoint_dir / "best"))
        new.best_checkpoint_dir.mkdir(exist_ok=True)

        new = replace(new, hyperparam_dir=(new.exp_dir / "hyperparams"))
        new.hyperparam_dir.mkdir(exist_ok=True)

        commit = str(self.git_repo.current_commit()) if self.git_repo is not None else ""
        new.metadata["commit"] = commit
        new.metadata["dirty_repo"] = dirty_repo
        new.metadata["description"] = description
        new.metadata["time"] = curr_time
        new.overwrite_metadata()

        return new

    def overwrite_metadata(self):
        with open(self.exp_dir / "metadata.json", "w") as fp:
            json.dump(self.metadata, fp)

    def save_hyperparams(self, info:dict):
        fnames = list(self.hyperparam_dir.glob("*.json"))
        if len(fnames) == 0:
            max_idx = 0
        else:
            max_idx = max([int(f.stem.split('.')[1]) for f in self.hyperparam_dir.glob("*.json")])
        fname = f"hyperaparams.{max_idx+1}.json"
        with open(self.hyperparam_dir / fname, "w") as fp:
            json.dump(info, fp)


    def flax_save_checkpoint(
        self,
        state: Any,
        step: int,
        metric: int, # Metric info to save with checkpoint, highest values kept
    ):
        """Saves the current step to the checkpoint_dir, saves the best checkpoint to the 'best' subdir"""
        did_save = False
        try:
            checkpoints.save_checkpoint(self.best_checkpoint_dir, state, metric, keep=self.keep_best)
            did_save = True
        except InvalidCheckpointError:
            did_save = False

        try:
            checkpoints.save_checkpoint(self.checkpoint_dir, state, step, keep=1)
            did_save=True
        except InvalidCheckpointError:
            did_save = False

        return did_save

    def maybe_save_checkpoint(
        self,
        state:Any, # Object to save. Can be any pytree
        step:int, # Step info to save with ckpt
        new_metric:Any, # Metric to pass to our comparator to determine if the checkpoint should be saved
        epoch:Optional[int]=None, # If provided, include epoch information in the ckpt
    ):
        savename = f"step-{step}_ckpt"
        did_save = False
        savename = f"step-{step}_{savename}"
        if epoch is not None:
            savename = f"epoch-{epoch}_{savename}"

        do_overwrite = (epoch is None) and (step is None)

        def save_model():
            self._ckpt_uid += 1
            checkpoints.save_checkpoint(self.checkpoint_dir, state, step, keep=self.keep_best)
            self._checkpoint_stack.append((value, savename))
            self._checkpoint_stack = sorted(self._checkpoint_stack)

        value = self.metric_calculator(new_metric)
        nckpts = len(self._checkpoint_stack)

        if nckpts < self.keep_best:
            save_model()
            did_save = True
        elif nckpts >= self.keep_best:
            prev_val, prev_fname = self._checkpoint_stack[-1]
            if value < prev_val:
                self._checkpoint_stack.pop(-1)
                save_model()
                # os.unlink(self.checkpoint_dir / prev_fname)
                did_save = True

        return did_save
