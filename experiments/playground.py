#!/usr/bin/env python3

BS=200
lr=1e-2
NUM_EPOCHS=1000

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


SS = PermutationSequence()
TT = TrivialSequence(SS.group_sequence())
S5 = SS + SS
NN = EMLPSequence(SS, # Rep in
                  TT, # Rep out
                  [] # Hidden layers
                  );
model = NN.emlp_at_level(7)
# model = emlp.nn.EMLP(SS.representation(7),TT.representation(7), group=S(7), num_layers=3,ch=35)
N = 10000

train_dataset = []
test_dataset = []
for j in range(N):
    x = np.random.rand(7)
    y = sum(x)
    train_dataset.append((x,y))

for j in range(N):
    x = np.random.rand(7)
    y = sum(x)
    test_dataset.append((x,y))

    

opt = objax.optimizer.Adam(model.vars())

@objax.Jit
@objax.Function.with_vars(model.vars())
def loss(x, y):
    yhat = model(x)
    return ((yhat-y)**2).mean()

grad_and_val = objax.GradValues(loss, model.vars())

@objax.Jit
@objax.Function.with_vars(model.vars()+opt.vars())
def train_op(x, y, lr):
    g, v = grad_and_val(x, y)
    opt(lr=lr, grads=g)
    return v

trainloader = DataLoader(train_dataset,batch_size=BS,shuffle=True)
testloader = DataLoader(test_dataset,batch_size=BS,shuffle=True)
print("It worked out!")


test_losses = []
train_losses = []
for epoch in tqdm(range(NUM_EPOCHS)):
    train_losses.append(np.mean([train_op(jnp.array(x),jnp.array(y),lr) for (x,y) in trainloader]))
    if not epoch%10:
        test_losses.append(np.mean([loss(jnp.array(x),jnp.array(y)) for (x,y) in testloader]))

import matplotlib.pyplot as plt
plt.plot(np.arange(NUM_EPOCHS),train_losses,label='Train loss')
plt.plot(np.arange(0,NUM_EPOCHS,10),test_losses,label='Test loss')
plt.legend()
plt.yscale('log')
v = np.random.randn(7)
        
