#!/usr/bin/env python3

BS = 500
lr = 1e-2
NUM_EPOCHS = 2000

import objax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from emlp.reps import (
    PermutationSequence,
    TrivialSequence,
    EquivariantOperatorSequence,
    null_space,
    lazify,
)
from emlp.nn import EMLPSequence
import emlp
import numpy as np
from emlp.groups import S
from objax.functional.loss import mean_squared_error

SS = PermutationSequence()
TT = TrivialSequence(SS.group_sequence())
V2 = SS * SS
# inner = V2 + V2 + V2 + V2 + V2
# inner = V2 + V2 + V2 + V2 + SS + SS + SS + SS + SS  # Two inner layers of this are good for l1 trace
inner = V2 + V2 + V2 + V2 + V2 + SS + SS + SS + SS + SS + SS + SS
NN = EMLPSequence(V2, TT, 2  * [inner], is_compatible=True)  # Rep in  # Rep out  # Hidden layers
d = 5
model = NN.emlp_at_level(d)


train_dataset = []
test_dataset = []
N = 2000
for j in range(N):
    x = np.random.randn(d, d)
    # y = sum(x)
    # y = np.sum(np.sqrt(np.abs(x)))
    # y = np.sum(np.abs(x))
    # y = (np.trace(np.matrix(x))
    # y = (np.trace(np.matrix(np.abs(x))))
    y = (np.sum(np.sqrt(np.diag(np.matrix(np.abs(x))))))
    train_dataset.append((x.reshape((d**2,)), y))

for j in range(N):
    x = np.random.randn(d, d)
    # y = sum(x)
    # y = np.sum(np.abs(x))
    # y = np.sum(np.sqrt(np.abs(x)))
    # y = (np.trace(np.matrix(np.abs(x))))
    y = (np.sum(np.sqrt(np.diag(np.matrix(np.abs(x))))))
    # y = np.sum(np.abs(x))
    test_dataset.append((x.reshape((d**2,)), y))


opt = objax.optimizer.Adam(model.vars())


@objax.Jit
@objax.Function.with_vars(model.vars())
def loss(x, y):
    yhat = model(x)
    return mean_squared_error(yhat.reshape(y.shape), y, 0)


grad_and_val = objax.GradValues(loss, model.vars())


@objax.Jit
@objax.Function.with_vars(model.vars() + opt.vars())
def train_op(x, y, lr):
    g, v = grad_and_val(x, y)
    opt(lr=lr, grads=g)
    return v, g


trainloader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=BS, shuffle=True)
print("Generated the data")


test_losses = []
train_losses = []
gradients = []
gra_n = []
for epoch in tqdm(range(NUM_EPOCHS)):
    losses = []
    gradient_norms = []
    for x, y in trainloader:
        v, g = train_op(jnp.array(x), jnp.array(y), lr)
        losses.append(v)
        gradients.append(g)
        # print(g))
    train_losses.append(np.mean(losses))
    gra_n.append(np.mean(gradient_norms))
    if not epoch % 10:
        test_losses.append(
            np.mean([loss(jnp.array(x), jnp.array(y)) for (x, y) in testloader])
        )
        print(f"Epoch {epoch} Train loss {train_losses[-1]} Test loss {train_losses[-1]} Grad norm {gra_n[-1]}")


import matplotlib.pyplot as plt

plt.plot(np.arange(NUM_EPOCHS), train_losses, label="Train loss")
plt.plot(np.arange(0, NUM_EPOCHS, 10), test_losses, label="Test loss")
plt.legend()
plt.yscale("log")
plt.show()
