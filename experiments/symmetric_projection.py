#!/usr/bin/env python3
import logging
from emlp.datasets import SymmetricProjection
from emlp.nn import EMLPSequence
from trainer.model_trainer import RegressorPlus
from utils import generate_datasets_in_dimensions, get_data_loaders
from oil.datasetup.datasets import split_dataset
import emlp.nn
import emlp.reps
import emlp.groups
from emlp.reps import PermutationSequence
import objax

seed = 926
ndata = 4000
batch_size = 600
num_epochs = 1000
step_size = lambda e: 2e-3
dimensions_to_extend = list(range(2, 11))
learning_dimension = 4
num_hidden_layers = 2


T1 = PermutationSequence()
seq_in = T1 * T1
inner = 4 * T1 + 4 * seq_in
seq_out = seq_in
free_NN = EMLPSequence(seq_in, seq_out, num_hidden_layers * [inner], is_compatible=True)
model = free_NN.emlp_at_level(learning_dimension)

log_level = "info"
split = {"train": -1, "val": 1000, "test": 1000}
trainer_config = {
    "log_dir": None,
    "log_args": {"minPeriod": 1, "timeFrac": 0.9},
    "early_stop_metric": "val_MSE",
}

datasets = generate_datasets_in_dimensions(
    SymmetricProjection, dimensions_to_extend, ndata, seed=seed
)
learning_datasets = split_dataset(datasets[learning_dimension], splits=split)
dataloaders = get_data_loaders(batch_size, learning_datasets)

solver = objax.optimizer.Adam
logging.getLogger().setLevel(logging.INFO)
trainer = RegressorPlus(model, dataloaders, solver, step_size, **trainer_config)
trainer.train(num_epochs)
