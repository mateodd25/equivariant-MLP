#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
from matplotlib.ticker import MaxNLocator


def extract_avg_min_max(data, method, l=None):
    chopped_data = [ser[:l] for ser in data[method]]
    return np.mean(chopped_data, 0), np.min(chopped_data, 0), np.max(chopped_data, 0)


def generate_figures(results_path, is_sine=False, use_legend=False):
    data = pickle.load(open(os.path.join(results_path, "state.p"), "rb"))
    dimensions = data["dimensions_to_extend"]
    avg_compatible, min_compatible, max_compatible = extract_avg_min_max(
        data["test_error_across_dim"], "compatible"
    )
    avg_free, min_free, max_free = extract_avg_min_max(
        data["test_error_across_dim"], "free"
    )
    learning_dimension = data["learning_dimension"]

    ticks = (
        dimensions if len(dimensions) < 10 else (dimensions[0 : len(dimensions) : 2])
    )

    with plt.style.context(["science"]):
        mpl.rcParams["xtick.minor.size"] = 0
        fig, ax = plt.subplots()
        ax.axvline(x=learning_dimension, color="darkgray", linestyle="dashdot")
        ax.plot(dimensions, avg_free, label="Free", color="#ffb000", linestyle="dashed")
        ax.fill_between(
            dimensions, min_free, max_free, color="#ffb000", linewidth=0.0, alpha=0.2
        )
        ax.plot(dimensions, avg_compatible, label="Compatible", color="#785ef0")
        ax.fill_between(
            dimensions,
            min_compatible,
            max_compatible,
            color="#785ef0",
            linewidth=0.0,
            alpha=0.2,
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.yscale("log")
        if is_sine:
            ppar = dict(xlabel=r"Dimension d$", ylabel=r"Sine squared error")
        else:
            ppar = dict(xlabel=r"Dimension $d$", ylabel=r"Mean squared error")
        if use_legend:
            ax.legend()
        ax.set(**ppar)
        ax.set_xticks(ticks)
        plt.savefig(os.path.join(results_path, "interdimensional_test_error.pdf"))

    l_compatible = min([len(l) for l in data["train_losses"]["compatible"]])
    avg_compatible, min_compatible, max_compatible = extract_avg_min_max(
        data["train_losses"], "compatible", l_compatible
    )
    l_free = min([len(l) for l in data["train_losses"]["free"]])
    avg_free, min_free, max_free = extract_avg_min_max(
        data["train_losses"], "free", l_free
    )

    with plt.style.context(["science"]):
        fig, ax = plt.subplots()
        ax.plot(
            range(l_free), avg_free, label="Free", color="#ffb000", linestyle="dashed"
        )
        ax.fill_between(
            range(l_free), min_free, max_free, color="#ffb000", linewidth=0.0, alpha=0.2
        )
        ax.plot(
            range(l_compatible), avg_compatible, label="Compatible", color="#785ef0"
        )
        ax.fill_between(
            range(l_compatible),
            min_compatible,
            max_compatible,
            color="#785ef0",
            linewidth=0.0,
            alpha=0.2,
        )
        plt.yscale("log")
        ppar = dict(xlabel=r"Iteration count", ylabel=r"Loss function")
        ax.legend()
        ax.set(**ppar)
        plt.savefig(os.path.join(results_path, "convergence.pdf"))


if __name__ == "__main__":
    results_path = "results/singular_vector/03:21PM-May-16-2023"
    generate_figures(results_path, is_sine=True)
    results_path = "results/O_invariance/12:30AM-May-17-2023/"
    generate_figures(results_path)
    results_path = "results/trace/12:59PM-May-16-2023/"
    generate_figures(results_path, use_legend=True)
    results_path = "results/diagonal_extraction/01:56AM-May-16-2023/"
    generate_figures(results_path)
    results_path = "results/symmetric_projection/10:13PM-May-15-2023/"
    generate_figures(results_path)
