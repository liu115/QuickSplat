from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, default_collate
from utils.utils import move_to_device


class GPUCacheLoader:
    # This is a Dataloader similar to the PyTorch DataLoader, but it loads all the data to the GPU for faster access.
    # This is useful for small datasets that can fit in the GPU memory.
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        device: str,
        verbose: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.data = []
        # Use tqdm to show the progress bar if verbose is True
        for i in tqdm(range(len(dataset)), disable=not verbose, desc="Loading data into GPU"):
            self.data.append(
                move_to_device(
                    dataset[i],
                    device,
                )
            )

    def _get_iterator(self):
        indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            data_batch = [self.data[j] for j in indices[i: i + self.batch_size]]
            # Stack the data to form a batch
            data_batch = default_collate(data_batch)
            yield data_batch

    def __iter__(self):
        return self._get_iterator()
