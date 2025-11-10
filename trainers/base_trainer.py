import json
import functools
from typing import Dict, List, Literal, Optional, Tuple, cast
import os
from pathlib import Path
import time

import wandb
from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from utils.optimizer import Optimizer
from utils.metrics import AverageMetric, Timer
from utils.scheduler import get_exponential_scheduler
from utils.vgg import VGGLoss
from utils.utils import move_to_device, step_check, check_main_thread, TimerContext
from utils.rich_utils import CONSOLE
from utils import comms


class BaseTrainer:
    PROJECT_NAME = "quicksplat"

    def __init__(
        self,
        config,
        local_rank,
        world_size,
        ckpt_path: Optional[str] = None,
        mode: Literal["train", "eval"] = "train",
    ):
        self.ckpt_path = ckpt_path
        assert mode in ["train", "eval"]
        self.mode = mode
        self.world_size = world_size
        self.local_rank = local_rank
        self.device: str = "cpu" if world_size == 0 else f"cuda:{local_rank}"

        # if ckpt_path is not None:
        #     if os.path.isdir(ckpt_path):
        #         files = sorted([x for x in os.listdir(ckpt_path) if x.endswith(".pth")])
        #         assert len(files) > 0, f"No checkpoint found in {ckpt_path}"
        #         ckpt_path = os.path.join(ckpt_path, files[-1])
        #         CONSOLE.print(f"Find config file {ckpt_path}")
        #     self.load_config_from_checkpoint(ckpt_path)
        #     CONSOLE.print(f"Loaded config from {ckpt_path}")
        #     print(self.config)
        # else:
        #     self.config = config
        self.config = config

        self._start_step = 1  # Global step
        self.setup_dataloader()

        self._model = self.get_model()
        self._model.to(self.device)
        if self.world_size > 1:
            self._model = DDP(self._model, device_ids=[local_rank], find_unused_parameters=False)
            CONSOLE.print("DDP initialized")
            dist.barrier(device_ids=[local_rank])

        self.setup_modules()

        self.setup_optimizers()

        self.setup_metrics()

        # self.setup_callbacks()

        self.setup_logger()

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)
            CONSOLE.print(f"Loaded checkpoint from {ckpt_path}")

    @property
    def model(self) -> nn.Module:
        """Returns the unwrapped model if in ddp"""
        if isinstance(self._model, DDP):
            return self._model.module
        return self._model

    def setup_dataloader(self):
        raise NotImplementedError

    def get_model(self) -> nn.Module:
        raise NotImplementedError

    def setup_modules(self):
        # Basic loss functions and metrics
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)
        self.vgg = VGGLoss().to(self.device)

    def train_step(self, batch, step_idx) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def train(self):
        self._model.train()

        step_idx = 0
        num_epoch = self.config.TRAIN.num_epochs
        grad_acc = self.config.TRAIN.grad_acc
        # assert grad_acc == 1, "[WIP] Gradient accumulation is not supported yet"
        if num_epoch == 0:
            num_iterations = self.config.TRAIN.num_iterations
            # Overriding num_epoch
            num_epoch = grad_acc * num_iterations // len(self.train_loader) + 1
        else:
            num_iterations = num_epoch * len(self.train_loader)
            num_epoch *= grad_acc
        print(f"Total number of iterations: {num_iterations}, epoch: {num_epoch}, start_step: {self._start_step}")

        # acc_step_cnt = 0
        if self.world_size > 1:
            dist.barrier()

        pbar = tqdm(total=num_iterations, desc="Training")
        if self._start_step > 1:
            pbar.update(self._start_step - 1)

        train_iter = iter(self.train_loader)
        self.num_iterations = num_iterations
        for step_idx in range(self._start_step, num_iterations + 1):
            self.iter_timer.start()
            pbar.update(1)

            for i in range(grad_acc):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                loss_dict, metrics_dict = self.train_step(batch, step_idx)
                loss = functools.reduce(torch.add, loss_dict.values())
                loss_dict["total_loss"] = loss

                with TimerContext("backward", self.config.debug):
                    (loss / grad_acc).backward()

            self.before_optimizer(step_idx)

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.train_metrics.update(metrics_dict)
            self.train_metrics.update(loss_dict)
            self.iter_timer.stop()

            self.after_train_step(step_idx)

            if step_idx % self.config.TRAIN.log_interval == 0:
                self.log_metrics(step_idx)

            if step_idx % self.config.TRAIN.val_interval == 0:
                save_dir = self.output_dir / "val"
                # Check main thread
                if self.local_rank == 0:
                    val_metrics = self.validate(step_idx, save_path=save_dir)
                    self.write_scalars("val", val_metrics, step_idx)

                    if self.config.save_checkpoint:
                        self.save_checkpoint(step_idx, val_metrics)
                else:
                    val_metrics = self.validate(step_idx, save_path=None)

            # if step_idx % self.config.TRAIN.save_interval == 0:
            #     self.save_checkpoint(step_idx)

            self.optimizer.step_scheduler()

            if self.world_size > 1:
                dist.barrier()
        pbar.close()

    def before_optimizer(self, step_idx: int):
        return

    def after_train_step(self, step_idx: int):
        return

    @check_main_thread
    def log_metrics(self, step_idx: int):
        if not self.config.do_logging:
            return

        self.writer.add_scalar("train/step_time", self.iter_timer.average(), step_idx)
        wandb.log({"train/step_time": self.iter_timer.average()}, step=step_idx)
        if hasattr(self, "optimizer"):
            if isinstance(self.optimizer, Optimizer):
                lr_dict = self.optimizer.get_lr_dict()
                for key, val in lr_dict.items():
                    self.writer.add_scalar(f"train/lr_{key}", val, step_idx)
                    wandb.log({f"train/lr_{key}": val}, step=step_idx)
        self.write_scalars("train", self.train_metrics.finalize(), step_idx)
        self.train_metrics.reset()
        self.iter_timer.reset()

    @check_main_thread
    def setup_logger(self):
        if self.mode == "train" and self.config.do_logging:
            log_dir = os.path.join(self.config.log_dir, self.config.name, self.config.subname)

            CONSOLE.log(f"Logging to {log_dir}")
            self.writer = SummaryWriter(log_dir=log_dir)
            self.writer.add_text("config", json.dumps(self.config, indent=4))

            if os.getenv("SLURM_JOBID"):
                self.writer.add_text("SLURM_JOBID", os.getenv("SLURM_JOBID"))

            # num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            # self.writer.add_text("num_params", f"{num_params:,}")

            wandb_name = self.config.name + "__" + self.config.subname
            if len(wandb_name) > 64:
                wandb_name = wandb_name[:64]
            wandb.init(
                project=self.PROJECT_NAME,
                dir=log_dir,
                name=wandb_name,
                # id=self.config.name + "__" + self.config.subname,
                notes=self.config.name + "__" + self.config.subname,
                reinit=True,
                config=self.config,
                resume="allow",
            )
            wandb.config.update(self.config, allow_val_change=True)
            if os.getenv("SLURM_JOBID"):
                wandb.log({"SLURM_JOBID": os.getenv("SLURM_JOBID")})

            # Write yaml config to self.out_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.output_dir / "config.yaml", "w") as f:
                import subprocess
                try:
                    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
                    f.write(f"# Commit hash: {commit_hash}\n")
                except:
                    pass
                f.write(self.config.dump())

    def setup_metrics(self):
        self.train_metrics = AverageMetric()
        self.iter_timer = Timer()
        self.change_timer = Timer()
        self.load_timer = Timer()
        self.inference_timer = Timer()

    def get_all_model_state_dict(self):
        return {
            "model": self.model.state_dict(),
        }

    @check_main_thread
    def save_checkpoint(self, step_idx: int, metrics_dict: Dict[str, float]) -> None:
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # save the checkpoint
        ckpt_path: Path = checkpoint_dir / f"step-{step_idx:09d}.ckpt"
        checkpoint = {
            "step": step_idx,
            **self.get_all_model_state_dict(),
        }
        if hasattr(self, "optimizer"):
            checkpoint["optimizer"] = self.optimizer.state_dict()
        torch.save(checkpoint, ckpt_path)
        CONSOLE.print(f"Saved checkpoint to {ckpt_path}")
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in checkpoint_dir.glob("*"):
                if f != ckpt_path:
                    f.unlink()

    def load_checkpoint(self, ckpt_path: str) -> None:
        if Path(ckpt_path).is_dir():
            ckpt_files = list(Path(ckpt_path).glob("*.ckpt"))
            assert len(ckpt_files) > 0, f"No checkpoint found in {ckpt_path}"
            ckpt_files.sort()
            ckpt_path = str(ckpt_files[-1])
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._start_step = ckpt["step"] + 1

    def setup_optimizers(self):

        self.optimizer = Optimizer(
            self.model.get_param_groups(),
            self.config.OPTIMIZER,
        )

    def write_scalars(self, tag, scalar_dict, step):
        if not self.config.do_logging:
            return

        for key, value in scalar_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{tag}/{key}", value, step)
                wandb.log({f"{tag}/{key}": value}, step=step)

    def move_to_device(self, batch):
        return move_to_device(batch, self.device)

    @property
    def output_dir(self) -> Path:
        return Path(self.config.save_dir) / self.config.name / self.config.subname

    @torch.no_grad()
    def validate(self, step_idx: int, save_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
