#!/usr/bin/env python3
import logging
import jax
from jax import device_put, jit
from .linear_operator_base import LinearOperator
import numpy as np
from tqdm.auto import tqdm
import jax.numpy as jnp
import optax
from emlp.utils import export


class ConvergenceError(Exception):
    """Iterative method did not converge."""

    pass


@export
def null_space(matrix: LinearOperator):
    """Return null space of a linear operator"""
    if matrix.shape[0] * matrix.shape[1] > 5e7:  # Too large to use SVD
        result = krylov_constraint_solve(matrix)
    else:
        dense_matrix = matrix.to_dense()
        result = orthogonal_complement(dense_matrix)
    return result


def orthogonal_complement(matrix):
    """Computes the orthogonal complement to a given matrix."""
    U, S, VH = jnp.linalg.svd(matrix, full_matrices=True)
    rank = (S > 1e-5).sum()
    return VH[rank:].conj().T


def krylov_constraint_solve(C, tol=1e-5):
    """Computes the solution basis Q for the linear constraint CQ=0  and QᵀQ=I
    up to specified tolerance with C expressed as a LinearOperator."""
    r = 5
    if C.shape[0] * r * 2 > 2e9:
        raise Exception(f"Solns for contraints {C.shape} too large to fit in memory")
    found_rank = 5
    while found_rank == r:
        r *= 2  # Iterative doubling of rank until large enough to include the full solution space
        if C.shape[0] * r > 2e9:
            logging.error(
                f"Hit memory limits, switching to sample equivariant subspace of size {found_rank}"
            )
            break
        Q = krylov_constraint_solve_upto_r(C, r, tol)
        found_rank = Q.shape[-1]
    return Q


def krylov_constraint_solve_upto_r(C, r, tol=1e-5, lr=1e-2):  # ,W0=None):
    """Iterative routine to compute the solution basis to the constraint CQ=0 and QᵀQ=I
    up to the rank r, with given tolerance. Uses gradient descent (+ momentum) on the
    objective |CQ|^2, which provably converges at an exponential rate."""
    W = np.random.randn(C.shape[-1], r) / np.sqrt(C.shape[-1])  # if W0 is None else W0
    W = device_put(W)
    opt_init, opt_update = optax.sgd(lr, 0.9)
    opt_state = opt_init(W)  # init stats

    def loss(W):
        return (
            jnp.absolute(C @ W) ** 2
        ).sum() / 2  # added absolute for complex support

    loss_and_grad = jit(jax.value_and_grad(loss))
    # setup progress bar
    pbar = tqdm(
        total=100,
        desc=f"Krylov Solving for Equivariant Subspace r<={r}",
        bar_format="{l_bar}{bar}| {n:.3g}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )
    prog_val = 0
    lstart, _ = loss_and_grad(W)

    for i in range(20000):
        lossval, grad = loss_and_grad(W)
        updates, opt_state = opt_update(grad, opt_state, W)
        W = optax.apply_updates(W, updates)
        # update progress bar
        progress = max(
            100 * np.log(lossval / lstart) / np.log(tol**2 / lstart) - prog_val, 0
        )
        progress = min(100 - prog_val, progress)
        if progress > 0:
            prog_val += progress
            pbar.update(progress)

        if jnp.sqrt(lossval) < tol:  # check convergence condition
            pbar.close()
            break  # has converged
        if lossval > 2e3 and i > 100:  # Solve diverged due to too high learning rate
            logging.warning(
                f"Constraint solving diverged, trying lower learning rate {lr/3:.2e}"
            )
            if lr < 1e-4:
                raise ConvergenceError(
                    f"Failed to converge even with smaller learning rate {lr:.2e}"
                )
            return krylov_constraint_solve_upto_r(C, r, tol, lr=lr / 3)
    else:
        raise ConvergenceError("Failed to converge.")
    # Orthogonalize solution at the end
    U, S, VT = np.linalg.svd(np.array(W), full_matrices=False)
    # Would like to do economy SVD here (to not have the unecessary O(n^2) memory cost)
    # but this is not supported in numpy (or Jax) unfortunately.
    rank = (S > 10 * tol).sum()
    Q = device_put(U[:, :rank])
    # final_L
    final_L = loss_and_grad(Q)[0]
    if final_L > tol:
        logging.warning(
            f"Normalized basis has too high error {final_L:.2e} for tol {tol:.2e}"
        )
    scutoff = S[rank] if r > rank else 0
    assert (
        rank == 0 or scutoff < S[rank - 1] / 100
    ), f"Singular value gap too small: {S[rank-1]:.2e} \
        above cutoff {scutoff:.2e} below cutoff. Final L {final_L:.2e}, earlier {S[rank-5:rank]}"
    # logging.debug(f"found Rank {r}, above cutoff {S[rank-1]:.3e} after {S[rank] if r>rank else np.inf:.3e}. Loss {final_L:.1e}")
    return Q
