#!/usr/bin/env python3
from emlp.nn import EMLPSequence
from objax.functional.loss import mean_squared_error
from tqdm.auto import tqdm
import objax
from jax import vmap
import jax.numpy as jnp
import gc
from torch.utils.data import DataLoader
from trainer.utils import LoaderTo
from oil.utils.utils import FixedNumpySeed, FixedPytorchSeed
from time import time
import numpy as np


def _norm_columns(x):
    xnorms = jnp.sum(x * x, 0)
    return xnorms


def angle_loss(yhat, y):
    return jnp.mean(
        1
        - jnp.sum(yhat * y, 0) ** 2 * (1 / _norm_columns(yhat) * (1 / _norm_columns(y)))
    )


def generate_datasets_across_dimensions(
    dimensions, random_sample, true_mapping, n=1024
):
    interdimensional_dataset = {}
    for i in dimensions:
        ext_test_data = []
        for _ in range(n):
            x = random_sample(i)
            ext_test_data.append((x, true_mapping(x)))
        interdimensional_dataset[i] = ext_test_data
    return interdimensional_dataset


def generate_datasets_fixed_dimension(
    dimension, random_sample, true_mapping, n=1024, nt=1024
):
    train_dataset = []
    test_dataset = []
    for j in range(n):
        x = random_sample(dimension)
        y = true_mapping(x)
        train_dataset.append((x.reshape(-1), y))

    for j in range(nt):
        x = random_sample(dimension)
        y = true_mapping(x)
        test_dataset.append((x.reshape(-1), y))

    return train_dataset, test_dataset


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


def test_different_dimensions(NN, dimensions_to_extend, test_data, loss):
    times = []
    mses = []
    equivariance_errors = []
    for i in dimensions_to_extend:
        ext_test_data = test_data[i]
        t1 = time()
        model = NN.emlp_at_level(i, trained=True)
        times.append(time() - t1)
        mse = []
        for x, y in ext_test_data:
            yhat = model(x.reshape(-1)).reshape(y.shape)
            mse.append(loss(yhat, y))
        mses.append(np.mean(mse))
        equivariance_errors.append(
            np.mean(
                [equivariance_err(model, x.reshape(-1), y) for x, y in ext_test_data]
            )
        )
        print(
            f"Level {i}, Time {times[-1]}, Test Error {mses[-1]}, Equiv Error {equivariance_errors[-1]}"
        )
        del model
        gc.collect()
    return times, mses, equivariance_errors


def test_different_dimensions_regression(NN, dimensions_to_extend, test_data):
    loss = lambda yhat, y: mean_squared_error(yhat, y)
    return test_different_dimensions(NN, dimensions_to_extend, test_data, loss)


def test_different_dimensions_angle(NN, dimensions_to_extend, test_data):
    return test_different_dimensions(NN, dimensions_to_extend, test_data, angle_loss)


def train_model(
    free_network,
    level,
    train_dataset,
    test_dataset,
    objective,
    dir_results=None,
    print_frequency=10,
    solver_config={
        "step_size": 6e-3,
        "num_epochs": 500,
        "batch_size": 500,
        "tolerance": 1e-10,
    },
):
    model = free_network.emlp_at_level(level)

    opt = objax.optimizer.Adam(model.vars())

    @objax.Jit
    @objax.Function.with_vars(model.vars())
    def loss(x, y):
        yhat = model(x)
        yhat = yhat.reshape(y.shape)
        return jnp.mean(objective(yhat, y))

    grad_and_val = objax.GradValues(loss, model.vars())

    @objax.Jit
    @objax.Function.with_vars(model.vars() + opt.vars())
    def train_op(x, y, lr):
        g, v = grad_and_val(x, y)
        opt(lr=lr, grads=g)
        return v, g

    trainloader = DataLoader(
        train_dataset, batch_size=solver_config["batch_size"], shuffle=True
    )
    testloader = DataLoader(
        test_dataset, batch_size=solver_config["batch_size"], shuffle=True
    )
    test_losses = []
    train_losses = []
    equiv_error = []
    for epoch in tqdm(range(solver_config["num_epochs"])):
        losses = []
        for x, y in trainloader:
            v, _ = train_op(jnp.array(x), jnp.array(y), solver_config["step_size"])
            losses.append(v)
        train_losses.append(np.mean(losses))
        if not epoch % print_frequency:
            test_losses.append(
                np.mean([loss(jnp.array(x), jnp.array(y)) for (x, y) in testloader])
            )
            equiv_error.append(
                np.mean(
                    [
                        equivariance_err(model, jnp.array(x), jnp.array(y))
                        for (x, y) in testloader
                    ]
                )
            )
            print(
                f"Epoch {epoch} Train loss {train_losses[-1]} Test loss {test_losses[-1]} Equi error {equiv_error[-1]}"
            )
        if train_losses[-1] < solver_config["tolerance"]:
            break

    free_network.set_trained_emlp_at_level(model)
    return free_network, train_losses, test_losses, equiv_error


def train_regression(
    free_network,
    level,
    train_dataset,
    test_dataset,
    dir_results=None,
    print_frequency=10,
    solver_config={
        "step_size": 6e-3,
        "num_epochs": 500,
        "batch_size": 500,
        "tolerance": 1e-10,
    },
):
    objective = lambda yhat, y: mean_squared_error(yhat.reshape(y.shape), y, None)
    return train_model(
        free_network,
        level,
        train_dataset,
        test_dataset,
        objective,
        dir_results=dir_results,
        print_frequency=print_frequency,
        solver_config=solver_config,
    )


def train_angle(
    free_network,
    level,
    train_dataset,
    test_dataset,
    dir_results=None,
    print_frequency=10,
    solver_config={
        "step_size": 6e-3,
        "num_epochs": 500,
        "batch_size": 500,
        "tolerance": 1e-10,
    },
):
    return train_model(
        free_network,
        level,
        train_dataset,
        test_dataset,
        angle_loss,
        dir_results=dir_results,
        print_frequency=print_frequency,
        solver_config=solver_config,
    )


def _train_and_test(
    compatible,
    seq_in,
    seq_out,
    inner,
    num_hidden_layers,
    use_gates,
    learning_dimension,
    train_set,
    test_set,
    solver_config,
    training_method,
    test_method,
    dimensions_to_extend,
    interdimensional_test_sets,
):
    NN_compatible = EMLPSequence(
        seq_in,
        seq_out,
        num_hidden_layers * [inner],
        is_compatible=compatible,
        use_gates=use_gates,
    )

    NN_compatible, train_loss, _, _ = training_method(
        NN_compatible,
        learning_dimension,
        train_set,
        test_set,
        solver_config=solver_config,
    )

    times, test_error, _ = test_method(
        NN_compatible,
        dimensions_to_extend,
        interdimensional_test_sets,
    )
    return train_loss, times, test_error


def generate_data_train_and_test(
    seed,
    dimensions_to_extend,
    random_sample,
    true_mapping,
    learning_dimension,
    n_train,
    n_test,
    seq_in,
    seq_out,
    inner,
    num_hidden_layers,
    solver_config,
    num_rep,
    use_gates=False,
    is_regression=True,
):
    train_losses = {"free": [], "compatible": []}
    times_to_extend = {"free": [], "compatible": []}
    test_error_across_dim = {"free": [], "compatible": []}
    training_method = train_regression if is_regression else train_angle
    test_method = (
        test_different_dimensions_regression
        if is_regression
        else test_different_dimensions_angle
    )
    with FixedNumpySeed(seed), FixedPytorchSeed(seed):
        for j in range(num_rep):
            interdimensional_test_sets = generate_datasets_across_dimensions(
                dimensions_to_extend,
                random_sample,
                true_mapping,
                n=n_test,
            )
            train_set, test_set = generate_datasets_fixed_dimension(
                learning_dimension,
                random_sample,
                true_mapping,
                n=n_train,
                nt=n_test,
            )

            train_loss, times, test_error = _train_and_test(
                True,  # Compatible?
                seq_in,
                seq_out,
                inner,
                num_hidden_layers,
                use_gates,
                learning_dimension,
                train_set,
                test_set,
                solver_config,
                training_method,
                test_method,
                dimensions_to_extend,
                interdimensional_test_sets,
            )
            train_losses["compatible"].append(train_loss)
            test_error_across_dim["compatible"].append(test_error)
            times_to_extend["compatible"].append(times)

            train_loss, times, test_error = _train_and_test(
                False,  # Compatible?
                seq_in,
                seq_out,
                inner,
                num_hidden_layers,
                use_gates,
                learning_dimension,
                train_set,
                test_set,
                solver_config,
                training_method,
                test_method,
                dimensions_to_extend,
                interdimensional_test_sets,
            )

            train_losses["free"].append(train_loss)
            test_error_across_dim["free"].append(test_error)
            times_to_extend["free"].append(times)

    return train_losses, times_to_extend, test_error_across_dim
