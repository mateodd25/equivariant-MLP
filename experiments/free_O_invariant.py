#!/usr/bin/env python3
import datetime
import pickle
import os
import numpy as np
import logging
from emlp.reps import OrthogonalSequence, TrivialSequence
from utils import (
    generate_data_train_and_test,
)
from generate_figures import generate_figures
from math import sin


def run_O_invariant_experiment():
    def random_sample(size):
        return np.random.rand(2 * size)

    def true_mapping(x):
        d = int(len(x) / 2)
        x1 = x[:d]
        x2 = x[d:]
        y = (
            sin(np.linalg.norm(x1))
            - np.linalg.norm(x2) ** 3 / 2
            + np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        )
        return y

    # Parameters
    logging.getLogger().setLevel(logging.INFO)
    seed = 2023
    n_train = 3000
    n_test = 1000
    dimensions_to_extend = list(range(2, 9))
    learning_dimension = 3
    num_rep = 3
    solver_config = {
        "step_size": 6e-3,
        # "num_epochs": 0,
        "num_epochs": 300,
        "batch_size": 500,
        "tolerance": 1e-8,
    }

    # Architecture
    T1 = OrthogonalSequence()
    T0 = TrivialSequence(T1.group_sequence())
    T2 = T1 * T1
    T3 = T2 * T1
    seq_in = T1 + T1
    # inner = 31 * T0 + 10 * T1 + 5 * T2 + 2 * T3
    inner = 10 * T0 + 5 * T1 + 5 * T2 + 3 * T3
    seq_out = T0
    num_hidden_layers = 2

    # Generate data, train, and test
    train_losses, times_to_extend, test_error_across_dim = generate_data_train_and_test(
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
        use_gates=True,
    )

    # Save and generate figures
    state = dict(
        seq_in=seq_in,
        inner=inner,
        seq_out=seq_out,
        seed=seed,
        n_train=n_train,
        n_test=n_test,
        dimensions_to_extend=dimensions_to_extend,
        learning_dimension=learning_dimension,
        num_hidden_layers=num_hidden_layers,
        num_rep=num_rep,
        solver_config=solver_config,
        train_losses=train_losses,
        times_to_extend=times_to_extend,
        test_error_across_dim=test_error_across_dim,
    )
    folder_name = datetime.datetime.now().strftime("%I:%M%p-%B-%d-%Y")
    results_directory = os.path.join("results", "O_invariance", folder_name)
    results_path = os.path.join(results_directory, "state.p")
    os.mkdir(results_directory)
    pickle.dump(state, open(results_path, "wb"))
    generate_figures(results_directory)
    print(f"The results were saved on: {results_directory}")


if __name__ == "__main__":
    run_O_invariant_experiment()
