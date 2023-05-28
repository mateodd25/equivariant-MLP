#!/usr/bin/env python3
import logging
from emlp.datasets import TraceData
from emlp.nn import EMLPSequence
from trainer.model_trainer import RegressorPlus
from utils import generate_datasets_in_dimensions, get_data_loaders
from oil.datasetup.datasets import split_dataset
import emlp.nn
import emlp.reps
import emlp.groups
from emlp.reps import PermutationSequence, TrivialSequence
import objax

seed = 926
ndata = 5000
batch_size = 500
num_epochs = 1000
step_size = lambda e: 1e-3
dimensions_to_extend = list(range(2, 11))
learning_dimension = 6
num_hidden_layers = 2


T1 = PermutationSequence()
T0 = TrivialSequence(T1.group_sequence())
seq_in = T1 * T1
inner = 4 * T1 + 4 * seq_in
seq_out = T0
free_NN = EMLPSequence(
    seq_in,
    seq_out,
    num_hidden_layers * [inner],
    use_bilinear=True,
    is_compatible=True,
    use_gates=False,
)
model = free_NN.emlp_at_level(learning_dimension)

log_level = "info"
split = {"train": -1, "val": 1000, "test": 1000}
trainer_config = {
    "log_dir": None,
    "log_args": {"minPeriod": 0.02, "timeFrac": 0.75},
    "early_stop_metric": "val_MSE",
}

datasets = generate_datasets_in_dimensions(
    TraceData, dimensions_to_extend, ndata, seed=seed
)
learning_datasets = split_dataset(datasets[learning_dimension], splits=split)
dataloaders = get_data_loaders(batch_size, learning_datasets)

solver = objax.optimizer.Adam
logging.getLogger().setLevel(logging.INFO)
trainer = RegressorPlus(model, dataloaders, solver, step_size, **trainer_config)
trainer.train(num_epochs)
