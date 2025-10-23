from typing import Dict
from contextlib import contextmanager
import time

import numpy as np
import torch


class AverageMetric:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.data = {}

    def update(self, loss_dict) -> None:
        for key, val in loss_dict.items():
            if val is None:
                continue
            if key not in self.data:
                self.data[key] = []
            if isinstance(val, torch.Tensor):
                val = val.item()
            self.data[key].append(val)

    def finalize(self) -> Dict[str, float]:
        loss_dict = {}
        for key, val in self.data.items():
            loss_dict[key] = np.mean(val)
        return loss_dict


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.record = []
        self.start_time = None
        self.start_pause_time = None
        self.total_pause_time = 0.0
        self.is_paused = False

    def start(self):
        self.start_time = time.time()

    def pause(self):
        if self.start_time is None:
            raise RuntimeError("Timer is not started. Please call start() first.")
        if self.is_paused:
            raise RuntimeError("Timer is already paused.")
        self.start_pause_time = time.time()
        self.is_paused = True

    def resume(self):
        if self.start_time is None:
            raise RuntimeError("Timer is not started. Please call start() first.")
        if not self.is_paused:
            raise RuntimeError("Timer is not paused.")
        self.total_pause_time += time.time() - self.start_pause_time
        self.is_paused = False
        self.start_pause_time = None

    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Timer is not started. Please call start() first.")
        if self.is_paused:
            # End the pause
            self.resume()

        end_time = time.time()
        total_time = end_time - self.start_time - self.total_pause_time
        self.record.append(total_time)
        self.start_time = None
        self.total_pause_time = 0.0
        # self.data.append(end_time - self.start_time)

    def current(self):
        if self.start_time is None:
            return 0.0
        # TODO: Consider the unfinish pause time
        # return time.time() - self.start_time - self.total_pause_time
        if not self.is_paused:
            return time.time() - self.start_time - self.total_pause_time
        else:
            pause_time = time.time() - self.start_pause_time
            return time.time() - self.start_time - self.total_pause_time - pause_time

    def average(self):
        if len(self.record) == 0:
            return 0.0
        return np.mean(self.record)

    # Pause context manager
    @contextmanager
    def pause_context(self):
        self.pause()
        yield
        self.resume()
