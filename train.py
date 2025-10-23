from __future__ import annotations

import os
import argparse
import random
from datetime import timedelta
from typing import Any, Callable, Literal, Optional
from tempfile import NamedTemporaryFile

import numpy as np
import torch
import torch.distributed as dist
from yacs.config import CfgNode as CN

from configs.config import get_cfg_defaults
from trainers import get_trainer
from utils.rich_utils import CONSOLE


DEFAULT_TIMEOUT = timedelta(minutes=30)


def parse_args():
    # Get config file path
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--scene_id", type=str, default=None, help="Scene ID for evaluation")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    # Other arguments (any input) that will be later merged into yacs config
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Modify config options using the command-line")
    return parser.parse_args()


def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True  # type: ignore


def launch(
    num_devices_per_machine: int,
    num_machines: int = 1,
    config: Optional[CN] = None,
    ckpt: Optional[str] = None,
) -> None:
    assert num_machines == 1, "Only single machine training is supported"
    if num_devices_per_machine * num_machines > 1:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group("nccl")
        local_rank = dist.get_rank()
        world_size = num_devices_per_machine
    else:
        local_rank = 0
        world_size = 1
    _set_random_seed(config.MACHINE.seed + local_rank)
    trainer = get_trainer(
        config,
        ckpt_path=ckpt,
        local_rank=local_rank,
        world_size=world_size,
        mode="train",
    )
    trainer.train()

    if num_devices_per_machine * num_machines > 1:
        dist.destroy_process_group()
    return


def main():
    args = parse_args()
    config = get_cfg_defaults()

    # Copy the args.config file to a temporary file
    with NamedTemporaryFile(mode="w", delete=True) as temp_config:
        with open(args.config, "r") as f:
            text = f.read()
            if args.scene_id is not None:
                text = text.replace("||SCENE_ID||", args.scene_id)
                print(text)
            temp_config.write(text)
            temp_config.flush()

        config.merge_from_file(temp_config.name)
        # config.merge_from_file(args.config)
        config.merge_from_list(args.opts)

        # Overwrite the config with debug mode
        if args.debug:
            config.debug = True
        if config.DATASET.overide_scene_id is not None:
            config.DATASET.overide_scene_id = str(config.DATASET.overide_scene_id)
        config.freeze()

        launch(
            num_devices_per_machine=config.MACHINE.num_devices,
            num_machines=config.MACHINE.num_machines,
            config=config,
            ckpt=args.ckpt,
        )


if __name__ == "__main__":
    main()


"""
In single GPU training, run the following command:
python train.py --config config.yaml

In multi-GPU training (e.g., 2 GPUs), run the following command:
torchrun --nnodes=1 --nproc_per_node=2 train.py --config config.yaml
"""
