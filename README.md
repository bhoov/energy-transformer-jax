# Energy Transformer

Submission code for ICML 2023

# Installation

We use conda to choose the `nvcc` and `cudnn` versions that work with our system (which uses A100 gpus) and JAX.

```
conda env create -f environment-dev.yml
conda activate energy-transformer
pip install -e .
pip install -r requirements.txt
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Test the install by starting python and running the following code:

```
import jax
print(jax.local_devices())
```

If you see GPU devices, great! If not, we need to change the cuda versions in the `environment-dev.yml` file to match what JAX supports on your system.

## Setting up data

The training code in this work assumes the ImageNet-1k Dataset in standard PyTorch ImageFolder structure. 

```
train/
    ...
    n02219486/
    n02487347/
    ...
val/
    ...
    n02219486/
    n02487347/
    ...
```

## See if it all works

Let's test the pretraining pipeline on imagenet1k with pytorch dataloading by default:

```
python scripts/train_et.py LOGS/my-experiment --train_data_dir=/PATH/TO/IN1K/TRAINING --val_data_dir=/PATH/TO/IN1K/VALIDATION 
```

If you have already run an experiment in `LOGS/my-experiment`, the saved checkpoint will be automatically loaded. If you want to ensure you are starting training from scratch, provide the `--no_autoresume` flag to ensure you have a clean `explogs/my-experiment` directory.

We can also extend to multiple GPU nodes by specifying a few additional flags:

```
python scripts/train_et.py LOGS/my-experiment --train_data_dir=/PATH/TO/IN1K/TRAINING --val_data_dir=/PATH/TO/IN1K/VALIDATION --server_ip "..." --server_port "..." --num_hosts X --host_idx Y
```

## Analyzing the Trained Model
A notebook in `nbs/Using-ET.ipynb` provides example code for loading the model, visualizing its learned parameters, and analyzing the image states through time.
