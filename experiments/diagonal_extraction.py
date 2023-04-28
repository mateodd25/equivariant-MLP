##!/usr/bin/env python3

import objax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from time import time
import gc
import pickle
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
import matplotlib.pyplot as plt
import scienceplots


def random_sample(size):
    return np.random.randn(size, size)


def to_evaluate(x):
    y = np.diag(np.diag(x))
    return y


def test_different_dimensions(NN, dimensions_to_extend, test_data):
    # models = []
    times = []
    mses = []
    j = 0
    for i in dimensions_to_extend:
        ext_test_data = test_data[j]
        j += 1
        t1 = time()
        model = NN.emlp_at_level(i, trained=True)
        times.append(time() - t1)
        mses.append(
            np.mean(
                [
                    (model(x.reshape(-1)).reshape(y.shape) - y) ** 2
                    for x, y in ext_test_data
                ]
            )
        )
        print(f"Level {i} time to extend {times[-1]} with MSE {mses[-1]}")
        del model
        gc.collect()
    return times, mses


if __name__ == "__main__":
    np.random.seed(926)
    BS = 500
    lr = 5e-3
    NUM_EPOCHS = 1000

    SS = PermutationSequence()
    V2 = SS * SS
    seq_in = V2
    inner = V2 + V2 + V2 + V2 + SS + SS + SS + SS
    seq_out = V2

    dimensions_to_extend = range(2, 11)
    interdimensional_test = []
    for i in dimensions_to_extend:
        ext_test_data = []
        for _ in range(100):
            x = random_sample(i)
            ext_test_data.append((x, to_evaluate(x)))
        interdimensional_test.append(ext_test_data)

    d = 4
    train_dataset = []
    test_dataset = []
    N = 2000
    for j in range(N):
        x = random_sample(d)
        y = to_evaluate(x)
        train_dataset.append((x.reshape((d**2,)), y))

    for j in range(N):
        x = random_sample(d)
        y = to_evaluate(x)
        test_dataset.append((x.reshape((d**2,)), y))

    def train_model(compatible):
        NN = EMLPSequence(
            seq_in, seq_out, 2 * [inner], is_compatible=compatible
        )  # Rep in  # Rep out  # Hidden layers
        model = NN.emlp_at_level(d)

        opt = objax.optimizer.Adam(model.vars())

        @objax.Jit
        @objax.Function.with_vars(model.vars())
        def loss(x, y):
            yhat = model(x)
            return mean_squared_error(yhat.reshape(y.shape), y, None)

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
                print(
                    f"Epoch {epoch} Train loss {train_losses[-1]} Test loss {train_losses[-1]} Grad norm {gra_n[-1]}"
                )

        NN.set_trained_emlp_at_level(model)
        return model, NN, train_losses, test_losses

    model_comp, NN_comp, train_losses_comp, test_losses_comp = train_model(True)
    times_comp, mses_comp = test_different_dimensions(
        NN_comp, dimensions_to_extend, interdimensional_test
    )

    model_free, NN_free, train_losses_free, test_losses_free = train_model(False)
    times_free, mses_free = test_different_dimensions(
        NN_free, dimensions_to_extend, interdimensional_test
    )

    with plt.style.context(["science", "vibrant"]):
        fig, ax = plt.subplots()
        ax.plot(dimensions_to_extend, mses_free, label="Free NN", linestyle="dashed")
        ax.plot(dimensions_to_extend, mses_comp, label="Compatible NN")
        plt.yscale("log")
        ppar = dict(xlabel=r"Dimension $d$", ylabel=r"Mean squared error")
        ax.legend()
        ax.set(**ppar)
        plt.savefig("diagonal_projection.pdf")

    state = dict(
        times_comp=times_comp,
        times_free=times_free,
        mses_comp=mses_comp,
        mses_free=mses_free,
    )
    pickle.dump(state, open("diagonal_projection_state.p", "wb"))
