#!/usr/bin/env python3
from scipy.sparse.linalg import svds
import datetime
import pickle
import os
import numpy as np
import logging
from emlp.reps import PermutationSequence, TrivialSequence
from utils import (
    generate_data_train_and_test,
)
from generate_figures import generate_figures


def run_singular_vector_experiment():
    def random_sample(size):
        return np.random.rand(size, size)

    def true_mapping(x):
        _, _, vh = svds(x, k=1)
        return vh.reshape(-1)

    # Parameters
    logging.getLogger().setLevel(logging.INFO)
    seed = 926
    n_train = 3000
    n_test = 1000
    dimensions_to_extend = list(range(2, 11))
    learning_dimension = 3
    num_rep = 3
    solver_config = {
        "step_size": 8e-3,
        "num_epochs": 300,
        "batch_size": 600,
        "tolerance": 1e-8,
    }

    # Architecture
    T1 = PermutationSequence()
    T0 = TrivialSequence(T1.group_sequence())
    T2 = T1 * T1
    T3 = T2 * T1
    seq_in = T2
    inner = 25 * T0 + 10 * T1 + 2 * T2 + T3
    seq_out = T1
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
        is_regression=False,  # Train with angle loss
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
    results_directory = os.path.join("results", "singular_vector", folder_name)
    results_path = os.path.join(results_directory, "state.p")
    os.mkdir(results_directory)
    pickle.dump(state, open(results_path, "wb"))
    generate_figures(results_directory)
    print(f"The results were saved on: {results_directory}")


if __name__ == "__main__":
    run_singular_vector_experiment()
