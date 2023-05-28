#!/usr/bin/env python3

import objax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from time import time
import gc
import pickle
from scipy.sparse.linalg import svds
from emlp.reps import (
    PermutationSequence,
    OrthogonalSequence,
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
import matplotlib.pyplot as plt
import scienceplots
from jax import vmap


def scale_adjusted_rel_err(a, b, g):
    return jnp.sqrt(((a - b) ** 2).mean()) / (
        jnp.sqrt((a**2).mean())
        + jnp.sqrt((b**2).mean())
        + jnp.abs(g - jnp.eye(g.shape[-1])).mean()
    )


def equivariance_err(model, x, y, group=None):
    try:
        model = model.model
    except:
        pass
    group = model.G if group is None else group
    gs = group.samples(x.shape[0])
    rho_gin = vmap(model.rep_in.rho_dense)(gs)
    rho_gout = vmap(model.rep_out.rho_dense)(gs)
    y1 = model((rho_gin @ x[..., None])[..., 0])
    y2 = (rho_gout @ model(x)[..., None])[..., 0]
    return np.asarray(scale_adjusted_rel_err(y1, y2, gs))


def random_sample(size):
    return np.random.rand(size, size)


def to_evaluate(x):
    _, _, vh = svds(x, k=1)
    return vh.reshape(-1)


def _norm_columns(x):
    xnorms = jnp.sum(x * x, 0)
    return xnorms


def test_different_dimensions(NN, dimensions_to_extend, test_data):
    # models = []
    times = []
    mses = []
    equi_error = []
    j = 0
    for i in dimensions_to_extend:
        ext_test_data = test_data[j]
        j += 1
        t1 = time()
        model = NN.emlp_at_level(i, trained=True)
        mses.append(
            [
                jnp.mean(
                    np.array(
                        [
                            1
                            - jnp.sum(model(x.reshape(-1)) * y, 0) ** 2
                            * (
                                1
                                / _norm_columns(model(x.reshape(-1)))
                                * (1 / _norm_columns(y))
                            )
                            for x, y in ext_test_data
                        ]
                    )
                )
            ]
        )
        times.append(time() - t1)
        equi_error.append(
            np.mean(
                [equivariance_err(model, x.reshape(-1), y) for x, y in ext_test_data]
            )
        )
        print(
            f"Level {i} time to extend {times[-1]} with MSE {mses[-1]} with Equi error {equi_error[-1]}"
        )
        del model
        gc.collect()
    return times, mses


BS = 600
lr = 8e-3
NUM_EPOCHS = 1000

T1 = PermutationSequence()
# T1 = OrthogonalSequence()
T0 = TrivialSequence(T1.group_sequence())
T2 = T1 * T1
T3 = T2 * T1
inner = 20 * T0 + 10 * T1 + 2 * T2 + T3
# inner = 4 * T0 + 4 * T1 + 2 * T2
num_inner_layers = 2


dimensions_to_extend = range(2, 11)
interdimensional_test = []
for i in dimensions_to_extend:
    ext_test_data = []
    for _ in range(100):
        x = random_sample(i)
        ext_test_data.append((x, to_evaluate(x)))
    interdimensional_test.append(ext_test_data)

d = 3
train_dataset = []
test_dataset = []
N = 3000
for j in range(N):
    x = random_sample(d)
    y = to_evaluate(x)
    train_dataset.append((x.reshape((d**2,)), y))

for j in range(100):
    x = random_sample(d)
    y = to_evaluate(x)
    test_dataset.append((x.reshape((d**2,)), y))


def train_model(compatible):
    NN = EMLPSequence(
        T2,
        T1,
        num_inner_layers * [inner],
        is_compatible=compatible,
        use_bilinear=True,
        use_gates=True,
    )  # Rep in  # Rep out  # Hidden layers
    model = NN.emlp_at_level(d)

    opt = objax.optimizer.Adam(model.vars())

    @objax.Jit
    @objax.Function.with_vars(model.vars())
    def loss(x, y):
        yhat = model(x)
        yhat = yhat.reshape(y.shape)
        return jnp.mean(
            1
            - jnp.sum(yhat * y, 0) ** 2
            * (1 / _norm_columns(yhat) * (1 / _norm_columns(y)))
        )

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
    # import pdb

    # pdb.set_trace()
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
            print(
                f"Epoch {epoch} Train loss {train_losses[-1]} Test loss {test_losses[-1]} Equi error {equivariance_err(model, jnp.array(x), jnp.array(y))}"
            )

    NN.set_trained_emlp_at_level(model)
    return model, NN, train_losses, test_losses


model_free, NN_free, train_losses_free, test_losses_free = train_model(False)
times_free, mses_free = test_different_dimensions(
    NN_free, dimensions_to_extend, interdimensional_test
)


model_comp, NN_comp, train_losses_comp, test_losses_comp = train_model(True)
times_comp, mses_comp = test_different_dimensions(
    NN_comp, dimensions_to_extend, interdimensional_test
)

with plt.style.context(["science", "vibrant"]):
    fig, ax = plt.subplots()
    ax.plot(dimensions_to_extend, mses_free, label="Free NN", linestyle="dashed")
    ax.plot(dimensions_to_extend, mses_comp, label="Compatible NN")
    plt.yscale("log")
    ppar = dict(xlabel=r"Dimension $d$", ylabel=r"Mean squared error")
    ax.legend()
    ax.set(**ppar)
    plt.savefig("InterdimensionalSVD.pdf")

    state = dict(
        times_comp=times_comp,
        times_free=times_free,
        mses_comp=mses_comp,
        mses_free=mses_free,
    )
    pickle.dump(state, open("state.p", "wb"))
