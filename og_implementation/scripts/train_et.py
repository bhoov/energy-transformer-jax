"""
```
python scripts/train_et.py PATH/TO/TRAINDATA PATH/TO/VALDATA llogs/final-days/prep-paper-largewdecay --task reconstruction --weight_decay 0.1
```
"""
from simple_parsing import ArgumentParser
from typing import *
from energy_transformer.flax_training_unified import TrainConfig, train_and_evaluate


parser = ArgumentParser()
parser.add_argument("workdir", type=str, help="Directory in which to save experiment metadata and checkpoints")

parser.add_arguments(TrainConfig, dest="config")
args = parser.parse_args()

if __name__ == "__main__":
    train_and_evaluate(args.workdir, args.config)
