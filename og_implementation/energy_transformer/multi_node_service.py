# Inspiration for code taken from https://raw.githubusercontent.com/yzhwang/jax-multi-gpu-resnet50-example/main/train.py
# And https://programtalk.com/vs4/python/EiffL/jax-gpu-cluster-demo/demo.py/

import atexit
import functools as ft
from typing import *
import os
import logging

from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_extension as xe
from jax._src.lib import xla_client as xc
from dataclasses import dataclass

@dataclass
class MultiNodeService:
    """Connect JAX to a multi-node system"""
    server_ip: str = '' # The IP address of the host on which to run the server.
    server_port: int = 5055 # The port on which to run the server
    num_hosts: int = 1 # The number of hosts (nodes) to synchronize.
    host_idx: int = 0 # The index of the host (node). Host 0 is the main host on which the server runs
    from_slurm: bool = False # If provided, overwrite the default 'ip' and 'server_port' information with defaults for slurm

    def __post_init__(self):
        if self.from_slurm:
            print(f"Overwriting default server config with SLURM env variables since `from_slurm` is True. Only `port={self.server_port}` is unchanged")
            self.host_idx = int(os.environ['SLURM_PROCID'])
            self.num_hosts = int(os.environ['SLURM_NTASKS'])
            self.server_ip = os.environ['SLURM_NODELIST'].split(',')[0].replace('[', '')



    @staticmethod
    def _reset_backend_state():
        xb._backends = {}
        xb._backends_errors = {}
        xb._default_backend = None
        xb.get_backend.cache_clear()

    def connect_to_gpu_cluster(self):
        self._reset_backend_state()
        service = None
        addr = f'{self.server_ip}:{self.server_port}'
        if self.host_idx == 0:
            # logging.info('starting service on %s', addr)
            print(f'starting service on {addr}')
            service = xe.get_distributed_runtime_service(addr, self.num_hosts)
            atexit.register(service.shutdown)

        print(f'connecting to service on {addr}')
        dist_client = xe.get_distributed_runtime_client(addr, self.host_idx)
        dist_client.connect()
        atexit.register(dist_client.shutdown)

        # register dist gpu backend
        factory = ft.partial(xc.make_gpu_client, dist_client, self.host_idx)
        xb.register_backend_factory('gpu', factory, priority=300)
        return service