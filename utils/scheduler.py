from typing import Optional, Literal
from dataclasses import dataclass

import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    # Backwards compatibility for PyTorch 1.x
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


@dataclass
class ExpSchedConfig:
    lr_pre_warmup: float = 1e-8
    lr_final: Optional[float] = None
    warmup_steps: int = 0
    max_steps: int = 100000
    ramp: Literal["linear", "cosine"] = "cosine"


def get_exponential_scheduler(
    optimizer: Optimizer,
    lr_init: float,
    config: ExpSchedConfig,
):
    if config.lr_final is None:
        lr_final = lr_init
    else:
        lr_final = config.lr_final

    def func(step):
        if step < config.warmup_steps:
            if config.ramp == "cosine":
                lr = config.lr_pre_warmup + (1 - config.lr_pre_warmup) * np.sin(
                    0.5 * np.pi * np.clip(step / config.warmup_steps, 0, 1)
                )
            else:
                lr = (
                    config.lr_pre_warmup
                    + (lr_init - config.lr_pre_warmup) * step / config.warmup_steps
                )
        else:
            t = np.clip(
                (step - config.warmup_steps) / (config.max_steps - config.warmup_steps),
                0,
                1,
            )
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return (
            lr / lr_init
        )  # divided by lr_init because the multiplier is with the initial learning rate

    scheduler = LambdaLR(optimizer, lr_lambda=func)
    return scheduler
