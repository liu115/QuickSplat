from typing import Dict, Any, List
import torch

from utils.scheduler import get_exponential_scheduler
from utils.rich_utils import CONSOLE


class Optimizer:

    def __init__(self, param_groups, config):
        self.config = config

        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.parameters: Dict[str, List[torch.nn.Parameter]] = {}
        self.optim_config: Dict[str, Any] = {}
        self.schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler] = {}

        for name, params in param_groups.items():
            param_config = getattr(config, name)
            if param_config is None:
                raise ValueError(f"Optimizer config for {name} is not found")
            params = list(params)
            if param_config.lr > 0 and len(list(params)) > 0:
                optimizer = self.get_optimizer(params, param_config)
                CONSOLE.print(f"Setup optimizer {name} with {param_config.optimizer_type} LR={param_config.lr} (total {sum(p.numel() for p in params):,} params)")
                scheduler = self.get_scheduler(optimizer, param_config)
                if scheduler is not None:
                    CONSOLE.print(f"Setup scheduler {name} {param_config.scheduler_type}")
                self.optimizers[name] = optimizer
                self.schedulers[name] = scheduler
                self.parameters[name] = params
                self.optim_config[name] = param_config

    def get_optimizer(self, params, config):
        if config.optimizer_type == "adam":
            optimizer = torch.optim.Adam(params, lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps, weight_decay=config.weight_decay)
        elif config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(params, lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps, weight_decay=config.weight_decay)
        elif config.optimizer_type == "rmsprop":
            optimizer = torch.optim.RMSprop(params, lr=config.lr, alpha=config.beta1, eps=config.eps, weight_decay=config.weight_decay)
        else:
            raise ValueError("Invalid optimizer type")
        return optimizer

    def get_scheduler(self, optimizer, config):
        if config.scheduler_type == "none":
            return None
        elif config.scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.step_size,
                gamma=config.gamma,
            )
        elif config.scheduler_type == "exp":
            return get_exponential_scheduler(
                optimizer,
                lr_init=config.lr,
                config=config,
            )
        else:
            raise ValueError("Invalid scheduler type")

    def step(self):
        for name, optimizer in self.optimizers.items():
            if self.optim_config[name].max_norm is not None:
                # print(f"======= Clipping grad norm {name} to {self.optim_config[name].max_norm} ======= ")
                # max_norm = 0.0
                # for params in self.parameters[name]:
                #     max_norm = max(max_norm, params.grad.view(-1).norm(2).item())
                # print(f"Max grad norm {name}: {max_norm:.4f}")
                torch.nn.utils.clip_grad_norm_(self.parameters[name], self.optim_config[name].max_norm)
                # max_norm = 0.0
                # for params in self.parameters[name]:
                #     max_norm = max(max_norm, params.grad.view(-1).norm(2).item())
                # print(f"Max grad norm clipped {name}: {max_norm:.4f}")
            optimizer.step()

    @torch.no_grad()
    def get_max_norm(self):
        norm_dict = {}
        for name, optimizer in self.optimizers.items():
            params = self.parameters[name]
            max_norm = 0.0
            for p in params:
                if p.grad is not None:
                    max_norm = max(max_norm, p.grad.view(-1).norm(2))
            if isinstance(max_norm, torch.Tensor):
                norm_dict[name] = max_norm.item()
            else:
                norm_dict[name] = max_norm
        return norm_dict

    def zero_grad(self):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

    def zero_grad_target(self, name):
        assert name in self.optimizers
        self.optimizers[name].zero_grad()

    def step_scheduler(self):
        for scheduler in self.schedulers.values():
            if scheduler is not None:
                scheduler.step()

    def state_dict(self):
        state_dict = {}
        state_dict["optimizers"] = {}
        for name, optimizer in self.optimizers.items():
            state_dict["optimizers"][name] = optimizer.state_dict()
        state_dict["schedulers"] = {}
        for name, scheduler in self.schedulers.items():
            if scheduler is not None:
                state_dict["schedulers"][name] = scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state_dict["optimizers"][name])
        for name, scheduler in self.schedulers.items():
            if scheduler is not None:
                scheduler.load_state_dict(state_dict["schedulers"][name])

    def get_lr_dict(self):
        lrs = {}
        for name, scheduler in self.schedulers.items():
            if scheduler is not None:
                lrs[name] = scheduler.get_last_lr()[0]
            else:
                lrs[name] = self.optimizers[name].param_groups[0]["lr"]
        return lrs
