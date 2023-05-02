import random

import numpy as np
import torch
import tempfile


def seed_all_random_engines(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


class CombinedDataLoader:
    def __init__(self, dataloader1, dataloader2):
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
        self.smallest_length = min(len(dataloader1), len(dataloader2))

    def __iter__(self):
        self.iterator1, self.iterator2 = (
            iter(self.dataloader1),
            iter(self.dataloader2),
        )
        self.counter = 0
        return self

    def __next__(self):
        if self.counter < self.smallest_length * 2:
            self.counter += 1

            # Choose a random dataloader
            choice = random.choice([0, 1])

            if choice == 0:
                return next(self.iterator1)
            else:
                return next(self.iterator2)
        else:
            raise StopIteration

    def __len__(self):
        return self.smallest_length * 2
