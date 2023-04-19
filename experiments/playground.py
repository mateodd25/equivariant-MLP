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
from emlp.reps import V
from objax.functional.loss import mean_squared_error

SS = PermutationSequence()
TT = TrivialSequence(SS.group_sequence())
S1 = SS * SS
S5 = S1 + S1
k = 2
NN = EMLPSequence(SS, # Rep in
                  TT, # Rep out
                  k * [S5] # Hidden layers
                  );
d = 5
# model = NN.emlp_at_level(d)

model = emlp.nn.EMLP(SS.representation(d),TT.representation(d), group=S(d), num_layers=k, ch=2 * (V ** 2))
# model = objax.nn.Linear(d, 1, use_bias=True)
N = 2000

train_dataset = []
test_dataset = []
for j in range(N):
    x = np.random.randn(d)
    # y = sum(x)
    # y = np.sum(np.sqrt(np.abs(x)))
    # y = np.sum(np.abs(x))
    y = np.sum(np.abs(x))
    train_dataset.append((x,y))

for j in range(N):
    x = np.random.randn(d)
    # y = sum(x)
    # y = np.sum(np.abs(x))
    # y = np.sum(np.sqrt(np.abs(x)))
    y = np.sum(np.abs(x))
    test_dataset.append((x,y))


# d = 8
# model = objax.nn.Linear(d, 1, use_bias=False)
# N = 100

# train_dataset = []
# test_dataset = []
# for j in range(N):
#     x = np.random.rand(7)
#     x = np.append(x, [1])
#     y = sum(x) - 1
#     train_dataset.append((x,y))

# for j in range(N):
#     x = np.random.rand(7)
#     x = np.append(x, [1])
#     y = sum(x) - 1
#     test_dataset.append((x,y))

    

opt = objax.optimizer.Adam(model.vars())

@objax.Jit
@objax.Function.with_vars(model.vars())
def loss(x, y):
    yhat = model(x).reshape(y.shape)
    return mean_squared_error(yhat, y, 0)

grad_and_val = objax.GradValues(loss, model.vars())

@objax.Jit
@objax.Function.with_vars(model.vars()+opt.vars())
def train_op(x, y, lr):
    g, v = grad_and_val(x, y)
    # print(f"Grad norm {np.linalg.norm(g)}")
    opt(lr=lr, grads=g)
    return v, g

trainloader = DataLoader(train_dataset,batch_size=BS,shuffle=True)
testloader = DataLoader(test_dataset,batch_size=BS,shuffle=True)
print("It worked out!")


test_losses = []
train_losses = []
gradients = []
gra_n = []
for epoch in tqdm(range(NUM_EPOCHS)):
    losses = []
    gradient_norms = []
    for (x, y) in trainloader:
        v, g = train_op(jnp.array(x),jnp.array(y),lr)
        losses.append(v)
        gradients.append(g)
        # gradient_norms.append(np.linalg.norm(g))
    train_losses.append(np.mean(losses))
    # gra_n.append(np.mean(gradient_norms))
    if not epoch%10:
        test_losses.append(np.mean([loss(jnp.array(x),jnp.array(y)) for (x,y) in testloader]))

import matplotlib.pyplot as plt
plt.plot(np.arange(NUM_EPOCHS),train_losses,label='Train loss')
plt.plot(np.arange(0,NUM_EPOCHS,10),test_losses,label='Test loss')
plt.legend()
plt.yscale('log')
v = np.random.randn(d)
        
