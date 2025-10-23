from typing import Any, Dict, Union, Callable
from copy import deepcopy
import time

import numpy as np
import cv2
import torch
import torch.distributed as dist

from . import comms


def move_to_device(batch, device: Union[torch.device, str] = "cuda"):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, dict):
        new_dict = {}
        for k, v in batch.items():
            new_dict[k] = move_to_device(v, device)
        return new_dict
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    else:
        return batch


def step_check(step, step_size, run_at_zero=False) -> bool:
    """Returns true based on current step and step interval."""
    if step_size == 0:
        return False
    return (run_at_zero or step != 0) and step % step_size == 0


def check_main_thread(func: Callable) -> Callable:
    """Decorator: check if you are on main thread"""

    def wrapper(*args, **kwargs):
        ret = None
        if comms.is_main_process():
            ret = func(*args, **kwargs)
        return ret

    return wrapper


def to_floats(data: Dict[str, Any]) -> Dict[str, float]:
    """Convert all values in a dictionary to floats."""
    out = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        else:
            v = float(v)
        out[k] = v
    return out


def timer_decorator(func: Callable) -> Callable:
    """Decorator: measure time of function"""

    def wrapper(*args, **kwargs):
        import time

        start = time.time()
        ret = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f} seconds")
        return ret

    return wrapper


global_timer_dict = {}


class TimerContext:
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.enabled:
            self.start = time.time()
            self.start_event.record()

    def __exit__(self, *args):
        if self.enabled:
            self.end_event.record()
            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    # print(f"{self.name} took {time.time() - self.start:.2f} seconds")

                    torch.cuda.synchronize()
                    print(f"{self.name} cuda event took {self.start_event.elapsed_time(self.end_event):.2f} ms")
            else:
                # print(f"{self.name} took {time.time() - self.start:.2f} seconds")

                torch.cuda.synchronize()
                spend = self.start_event.elapsed_time(self.end_event)
                # print(f"{self.name} cuda event took {spend:.2f} ms")
                global global_timer_dict
                if self.name in global_timer_dict:
                    global_timer_dict[self.name].append(spend)
                else:
                    global_timer_dict[self.name] = [spend]


def find_unique_indices(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Find the first indices of unique elements in a tensor. Similar to np.unique(return_index=True)"""
    unique, idx, counts = torch.unique(x, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=x.device), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    return first_indicies


def write_text_on_image(image, text: str):
    # Place the text on the image top-left with black color.
    image = np.ascontiguousarray(image)
    image = cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return image


def load_state_dict_custom(model, state_dict):
    # Allow loading state dict with different keys or same keys with different shapes
    model_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                new_state_dict[k] = v
            else:
                print(f"Skipping {k} due to shape mismatch")
        else:
            print(f"Skipping {k} due to missing key")

    return model.load_state_dict(new_state_dict, strict=False)
