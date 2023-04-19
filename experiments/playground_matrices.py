#!/usr/bin/env python3

BS=20
lr=5e-3
NUM_EPOCHS=2000

import objax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from emlp.reps import PermutationSequence, TrivialSequence, EquivariantOperatorSequence, null_space, lazify
from emlp.nn import EMLPSequence
import emlp
import numpy as np
from emlp.groups import S
from objax.functional.loss import mean_squared_error

SS = PermutationSequence()
TT = TrivialSequence(SS.group_sequence())
V2 = SS * SS
S5 = V2 + V2 + V2
NN = EMLPSequence(V2, # Rep in
                  V2, # Rep out
                  2 * [V2] # Hidden layers
                  );

d = 5
model = NN.emlp_at_level(d)


train_dataset = []
test_dataset = []
N = 100
for j in range(N):
    x = np.random.randn(d,d)
    # y = sum(x)
    # y = np.sum(np.sqrt(np.abs(x)))
    # y = np.sum(np.abs(x))
    y = np.trace(x)
    train_dataset.append((x.reshape((d ** 2,)),y))

for j in range(N):
    x = np.random.randn(d)
    # y = sum(x)
    # y = np.sum(np.abs(x))
    # y = np.sum(np.sqrt(np.abs(x)))
    y = np.trace(x)
    # y = np.sum(np.abs(x))
    train_dataset.append((x.reshape((d ** 2,)),y))

