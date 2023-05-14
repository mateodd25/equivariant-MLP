#!/usr/bin/env python3
from torch.utils.data import DataLoader
from trainer.utils import LoaderTo
from oil.utils.utils import FixedNumpySeed, FixedPytorchSeed


def generate_datasets_in_dimensions(dataset, dimensions, n=1024, seed=926):
    """Generates #dimensions datasets, one per each dimension in dimensions."""
    datasets = {}
    with FixedNumpySeed(seed), FixedPytorchSeed(seed):
        for d in dimensions:
            datasets[d] = dataset(N=n, dimension=d)
    return datasets


def get_data_loaders(batch_size, datasets):
    return {
        k: LoaderTo(
            DataLoader(
                v,
                batch_size=min(batch_size, len(v)),
                shuffle=(k == "train"),
                num_workers=0,
                pin_memory=False,
            )
        )
        for k, v in datasets.items()
    }
