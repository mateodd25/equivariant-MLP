import jax
import jax.numpy as jnp
import objax.nn as nn
import objax.functional as F
import numpy as np
from emlp.reps import T, Rep, Scalar
from emlp.reps import EquivariantOperatorSequence, EquivariantOperators
from emlp.reps import bilinear_weights
from emlp.reps.product_sum_reps import SumRep
import collections
from emlp.utils import Named, export
import scipy as sp
import scipy.special
import random
import logging
from objax.variable import TrainVar, StateVar
from objax.nn.init import kaiming_normal, xavier_normal
from objax.module import Module
import objax
from objax.nn.init import orthogonal
from scipy.special import binom
from jax import jit, vmap
from functools import lru_cache as cache
from objax.util import class_name



def Sequential(*args):
    """Wrapped to mimic pytorch syntax"""
    return nn.Sequential(args)


@export
class Linear(nn.Linear):
    """Basic equivariant Linear layer from repin to repout."""

    def __init__(self, repin, repout):
        nin, nout = repin.size(), repout.size()
        super().__init__(nin, nout)
        self.b = TrainVar(objax.random.uniform((nout,)) / jnp.sqrt(nout))
        self.w = TrainVar(orthogonal((nout, nin)))
        self.rep_W = rep_W = repout * repin.T

        rep_bias = repout
        self.Pw = rep_W.equivariant_projector()
        self.Pb = rep_bias.equivariant_projector()
        logging.info(f"Linear W components:{rep_W.size()} rep:{rep_W}")

    def __call__(self, x):  # (cin) -> (cout)
        logging.debug(f"linear in shape: {x.shape}")
        W = (self.Pw @ self.w.value.reshape(-1)).reshape(*self.w.value.shape)
        b = self.Pb @ self.b.value
        out = x @ W.T + b
        logging.debug(f"linear out shape:{out.shape}")
        return out


@export
class BiLinear(Module):
    """Cheap bilinear layer (adds parameters for each part of the input which can be
    interpreted as a linear map from a part of the input to the output representation).
    """

    def __init__(self, repin, repout):
        super().__init__()
        Wdim, weight_proj = bilinear_weights(repout, repin)
        self.weight_proj = jit(weight_proj)
        self.w = TrainVar(objax.random.normal((Wdim,)))
        logging.info(f"BiW components: dim:{Wdim}")

    def __call__(self, x, training=True):
       # compatible with non sumreps? need to check
        W = self.weight_proj(self.w.value, x)
        out = 0.1 * (W @ x[..., None])[..., 0]
        return out


@export
def gated(ch_rep: Rep) -> Rep:
    """Returns the rep with an additional scalar 'gate' for each of the nonscalars and non regular
    reps in the input. To be used as the output for linear (and or bilinear) layers directly
    before a :func:`GatedNonlinearity` to produce its scalar gates."""
    if isinstance(ch_rep, SumRep):
        return ch_rep + sum(
            [
                Scalar(rep.G)
                for rep in ch_rep
                if rep != Scalar and not rep.is_permutation
            ]
        )
    else:
        return ch_rep + Scalar(ch_rep.G) if not ch_rep.is_permutation else ch_rep


@export
class GatedNonlinearity(Module):
    """Gated nonlinearity. Requires input to have the additional gate scalars
    for every non regular and non scalar rep. Applies swish to regular and
    scalar reps."""

    def __init__(self, rep):
        super().__init__()
        self.rep = rep

    def __call__(self, values):
        gate_scalars = values[..., gate_indices(self.rep)]
        activations = jax.nn.relu(gate_scalars) * values[..., : self.rep.size()]
        return activations

@export
class ReluNonlinearity(Module):
    """Gated nonlinearity. Requires input to have the additional gate scalars
    for every non regular and non scalar rep. Applies swish to regular and
    scalar reps."""

    def __init__(self, rep):
        super().__init__()
        self.rep = rep

    def __call__(self, values):
        activations = jax.nn.relu(values)
        return activations


@export
class EMLPBlock(Module):
    """Basic building block of EMLP consisting of G-Linear, biLinear,
    and gated nonlinearity."""

    def __init__(self, rep_in, rep_out):
        super().__init__()
        self.linear = Linear(rep_in, gated(rep_out))
        self.bilinear = BiLinear(gated(rep_out), gated(rep_out))
        self.nonlinearity = GatedNonlinearity(rep_out)

    def __call__(self, x):
        lin = self.linear(x)
        preact = self.bilinear(lin) + lin
        return self.nonlinearity(preact)


def uniform_rep_general(ch, *rep_types):
    """adds all combinations of (powers of) rep_types up to
    a total size of ch channels."""
    raise NotImplementedError


@export
def uniform_rep(ch, group):
    """A heuristic method for allocating a given number of channels (ch)
    into tensor types. Attempts to distribute the channels evenly across
    the different tensor types. Useful for hands off layer construction.

    Args:
        ch (int): total number of channels
        group (Group): symmetry group

    Returns:
        SumRep: The direct sum representation with dim(V)=ch
    """
    d = group.d
    Ns = np.zeros((lambertW(ch, d) + 1,), int)  # number of tensors of each rank
    while ch > 0:
        max_rank = lambertW(ch, d)  # compute the max rank tensor that can fit up to
        Ns[: max_rank + 1] += np.array(
            [d ** (max_rank - r) for r in range(max_rank + 1)], dtype=int
        )
        ch -= (max_rank + 1) * d**max_rank  # compute leftover channels
    sum_rep = sum([binomial_allocation(nr, r, group) for r, nr in enumerate(Ns)])
    sum_rep, perm = sum_rep.canonicalize()
    return sum_rep


def lambertW(ch, d):
    """Returns solution to x*d^x = ch rounded down."""
    max_rank = 0
    while (max_rank + 1) * d**max_rank <= ch:
        max_rank += 1
    max_rank -= 1
    return max_rank


def binomial_allocation(N, rank, G):
    """Allocates N of tensors of total rank r=(p+q) into
    T(k,r-k) for k=0,1,...,r to match the binomial distribution.
    For orthogonal representations there is no
    distinction between p and q, so this op is equivalent to N*T(rank)."""
    if N == 0:
        return 0
    n_binoms = N // (2**rank)
    n_leftover = N % (2**rank)
    even_split = sum(
        [n_binoms * int(binom(rank, k)) * T(k, rank - k, G) for k in range(rank + 1)]
    )
    ps = np.random.binomial(rank, 0.5, n_leftover)
    ragged = sum([T(int(p), rank - int(p), G) for p in ps])
    out = even_split + ragged
    return out


def uniform_allocation(N, rank):
    """Uniformly allocates N of tensors of total rank r=(p+q) into
    T(k,r-k) for k=0,1,...,r. For orthogonal representations there is no
    distinction between p and q, so this op is equivalent to N*T(rank)."""
    if N == 0:
        return 0
    even_split = sum((N // (rank + 1)) * T(k, rank - k) for k in range(rank + 1))
    ragged = sum(
        random.sample([T(k, rank - k) for k in range(rank + 1)], N % (rank + 1))
    )
    return even_split + ragged


@export
class ExtendableLinear(nn.Linear):
    """Extendable equivariant linear layer."""

    def __init__(
        self,
        repin,
        repout,
        include_bias=True,
        compatibility_constraints=None,
        learned_parameters = None,
    ):
        self.size_in, self.size_out = repin.size(), repout.size()
        super().__init__(self.size_in, self.size_out, use_bias=include_bias)

        # If the parameters are already learned, there is no need to use a basis
        if learned_parameters is not None:
            self.use_basis = False
            w, b = learned_parameters
            self.w = TrainVar(w)
            if b is not None and include_bias:
                self.b = TrainVar(b)
            if include_bias and b is None:
                raise ValueError("Cannot initialize the bias as None when passing include_bias=True")

        else:
            self.use_basis = True
            self.use_bias = include_bias
            if include_bias:
                self.bias_basis = repout.equivariant_basis()
                self.bias_size = self.bias_basis.shape[1]
                self.b = TrainVar(
                    objax.random.uniform((self.bias_size,)) / jnp.sqrt(1000)
                    )
                print("Use bias")
            else:
                print("No bias")
                self.bias_basis = None
            # TODO: Fix this hack once EquivariantLinearMaps is implemented properly. (Receive basis as oppposed to compatibility conditions)
            if compatibility_constraints is not None:
                self.rep_W = rep_W = EquivariantOperators(
                    repin, repout, compatibility_constraints
                )  # TODO: Make class to handle this?
                self.basis = rep_W.equivariant_basis()
                print("Compatible")
            else:
                self.rep_W = rep_W = repin >> repout
                self.basis = rep_W.equivariant_basis()
                print("Not compatatible")
            basis_size = self.basis.shape[1]

            if coeff_val is not None:
                self.w = TrainVar(coeff_val)
            else:
                self.w = TrainVar(
                    objax.random.uniform((basis_size,)) / jnp.sqrt(1000)
                )

    def __call__(self, x):
        logging.debug(f"Linear in shape: {x.shape}")
        if use_basis:
            W = (self.basis @ self.w).reshape((self.size_out, self.size_in))
            out = (x @ W.T)
            if self.use_bias:
                out = out + (self.bias_basis @ self.b)
            logging.debug(f"Linear out shape: {out.shape}")
        else:
            out = x @ self.w.T
            if self.use_bias:
                out = out + self.b
        return out

    def __repr__(self):
        args = f"nin={self.size_in}, nout={self.size_out}, use_bias={self.use_bias}, use_basis={self.use_basis}"
        return f"{class_name(self)}({args})"

    def get_linear_map(self):
        if self.use_basis:
            return (self.basis @ self.w).reshape((self.size_out, self.size_in))
        return self.w

    def get_bias(self):
        if self.use_bias is False:
            raise ValueError("Layer does not use bias.")
        if self.use_basis:
            return (self.bias_basis @ self.b)
        return self.b



@export
class ExtendableEMLPBlock(Module):
    def __init__(self, rep_in, rep_out, compatibility_constraints=None, use_bias=True, learned_parameters=None):
        super().__init__()
        self.linear = ExtendableLinear(
            rep_in,
            rep_out,
            include_bias=use_bias,
            compatibility_constraints=compatibility_constraints,
            learned_parameters=learned_parameters,
        )
        # TODO: Implement Bilinear
        # self.bilinear = ExtendableBilinear(rep_out, rep_out)
        # self.nonlinearity = GatedNonlinearity(rep_out)
        self.nonlinearity = ReluNonlinearity(rep_out)

    def __call__(self, x):
        lin = self.linear(x)
        return self.nonlinearity(lin)


@export
class ExtendableEMLP(Module, metaclass=Named):
    """
    Given a sequence of architectures. It creates an EMLP at level k.

    The argument is_compatible determines whether the EMLP satisfies compatibility conditions.
    If learned_parameters = {(W_i, b_i)} is not null it initializes the layers with these parameters.
    """

    def __init__(
            self, sequence_in,
            sequence_out, hidden_sequences, k, is_compatible=False, use_bias=True, learned_parameters=None
    ):
        self.rep_in = sequence_in.representation(k)
        self.rep_out = sequence_out.representation(k)
        self.G = self.rep_in.G
        self.is_compatible = is_compatible
        self.level = k

        if learned_parameters is not None:
            # TODO: Add checks to ensure that he learned parameters make sense for the given sequences
            self.use_basis = False
            sequences = [sequence_in] + hidden_sequences
            j = 0
            for sin, sout in zip(sequences, sequences[1:]):
                layers.append(
                    ExtendableEMLPBlock(
                        sin.representation(k),
                        sout.representation(k),
                        compatibility_constraints=EquivariantOperatorSequence(
                            sin, sout
                        ).compatibility_constraints(k),
                        use_bias=use_bias,
                        learned_parameters=learned_parameters[j]
                    )
                )
                j += 1
            self.network = Sequential(
                *layers,
                ExtendableLinear(
                    sequences[-1].representation(k),
                    self.rep_out,
                    include_bias = use_bias,
                    learned_parameters = learned_parameters[-1]
                )
            )

        else:
            self.use_basis = True
            if is_compatible:
                sequences = [sequence_in] + hidden_sequences
                layers = []
                for sin, sout in zip(sequences, sequences[1:]):
                    layers.append(
                        ExtendableEMLPBlock(
                            sin.representation(k),
                            sout.representation(k),
                            compatibility_constraints=EquivariantOperatorSequence(
                                sin, sout
                            ).compatibility_constraints(k),
                            use_bias=use_bias,
                        )
                    )

                logging.info(f"Sequences: {sequences}")
                self.network = Sequential(
                    *layers,
                    ExtendableLinear(
                        sequences[-1].representation(k),
                        self.rep_out,
                        include_bias=use_bias,
                        compatibility_constraints=EquivariantOperatorSequence(
                                sequences[-1], sequence_out
                        ).compatibility_constraints(k),
                    ),
                )
            else:
                reps = [self.rep_in] + [
                    sequence.representation(k) for sequence in hidden_sequences
                ]
                logging.info(f"Reps: {reps}")
                self.network = Sequential(
                    *[ExtendableEMLPBlock(rin, rout) for rin, rout in zip(reps, reps[1:])],
                    ExtendableLinear(reps[-1], self.rep_out),
                )

    def __call__(self, x, training=True):
        return self.network(x)


@export
class EMLPSequence(object):
    """Lazy sequence of EMLP.

    Contains a sequence of EMLP. Might be used to instantiate an EMLP
    at a fixed level k.
    """

    def __init__(
            self,
            sequence_in,
            sequence_out,
            hidden_sequences,
            is_compatible=False
    ):
        self.is_trained = False
        self.trained_level = -1  # Stores the level at which the EMLP was trained
        self.learned_emlp = None
        self.sequence_in = sequence_in
        self.sequence_out = sequence_out
        self.hidden_sequences = hidden_sequences
        self.is_compatible = is_compatible
        self.use_bias = not is_compatible

    # TODO: Store a cached version of already evaluated levels, so as to not repeat computations.
    def emlp_at_level(self, j, trained=False):
        if self.is_trained is False and trained is True:
            raise ValueError(
                "At least one level of the EMLP sequence has to be trained before it can return trained EMLPs at any level."
            )

        # TODO Continue here
        # Extend when the EMLPSequence is already trained
        if self.is_trained:
            sequences = [self.sequence_in] + self.hidden_sequences + [self.sequence_out]
            layers_at_level = []
            learned_parameters = []
            for i, layer in enumerate(self.learned_emlp.network):
                seq_in = sequences[i]
                seq_out = sequences[i+1]

                if isinstance(layer, EMLPBlock):
                    W = layer.get_linear_map().reshape(-1)
                    constraints = EquivariantOperatorSequence(seq_in, seq_out).extendability_constraints(j, self.trained_level)
                    num_zeros = constratints.shape[0] - len(W)
                    right_hand_side = jnp.zeros(constraints.shape[0])
                    right_hand_side[-len(W):] = W
                    W_at_level, _ = jax.scipy.sparse.linalg.gmres(constraints, right_hand_side)
                    b_at_level = None

                    if self.use_bias:
                        b = layer.get_bias()
                        constraints_b = seq_in.representation(j).extendability_constraints(j, self.trained_level)
                        right_hand_side_b = jnp.zeros(constraints_b.shape[0])
                        num_zeros_b = constraints_b.shape[0] - len(b)
                        right_hand_side_b = jnp.zeros(constraints_b.shape[0])
                        b_at_level, _ = jax.scipy.sparse.linalg.gmres(constraints_b, right_hand_side_b)

                    learned_parameters.append((W_at_level.reshape(seq_out.dimension(j), seq_in.dimension(j)), b_at_level))


                elif isinstance(layer, ExtendableLinear):

                else:
                    raise ValueError(f"Cannot extend layer {layer}")


        # TODO: Make it work once things are trained
        # # If the sequence has not been trained, return a randomly initialized EMLP
        return ExtendableEMLP(e
            self.sequence_in, self.sequence_out, self.hidden_sequences,
            j,  # Level
            is_compatible=self.is_compatible,
            use_bias=(not self.is_compatible)
        )

    def set_trained_emlp_at_level(self, emlp: ExtendableEMLP):
        if self.is_trained:
            raise ValueError("A trained emlp cannot be set again.")
        self.trained_level = emlp.level
        self.learned_emlp = emlp
        self.is_trained = True

@export
class EMLP(Module, metaclass=Named):
    """Equivariant MultiLayer Perceptron.

    If the input ch argument is an int, uses the hands off uniform_rep heuristic.
    If the ch argument is a representation, uses this representation for the hidden layers.
    Individual layer representations can be set explicitly by using a list of ints or a list of
    representations, rather than use the same for each hidden layer.

    Args:
        rep_in (Rep): input representation
        rep_out (Rep): output representation
        group (Group): symmetry group
        ch (int or list[int] or Rep or list[Rep]): number of channels in the hidden layers
        num_layers (int): number of hidden layers

    Returns:
        Module: the EMLP objax module."""

    def __init__(self, rep_in, rep_out, group, ch=384, num_layers=3):  # @
        super().__init__()
        logging.info("Initing EMLP (objax)")
        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)

        self.G = group
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch, int):
            middle_layers = num_layers * [
                uniform_rep(ch, group)
            ]  # [uniform_rep(ch,group) for _ in range(num_layers)]
        elif isinstance(ch, Rep):
            middle_layers = num_layers * [ch(group)]
        else:
            middle_layers = [
                (c(group) if isinstance(c, Rep) else uniform_rep(c, group)) for c in ch
            ]
        # assert all((not rep.G is None) for rep in middle_layers[0].reps)
        reps = [self.rep_in] + middle_layers
        logging.info(f"Reps: {reps}")
        self.network = Sequential(
            *[EMLPBlock(rin, rout) for rin, rout in zip(reps, reps[1:])],
            Linear(reps[-1], self.rep_out),
        )

    def __call__(self, x, training=True):
        return self.network(x)


def swish(x):
    return jax.nn.sigmoid(x) * x


def MLPBlock(cin, cout):
    return Sequential(
        nn.Linear(cin, cout), swish
    )  # ,nn.BatchNorm0D(cout,momentum=.9),swish)#,


@export
class MLP(Module, metaclass=Named):
    """Standard baseline MLP. Representations and group are used for shapes only."""

    def __init__(self, rep_in, rep_out, group, ch=384, num_layers=3):
        super().__init__()
        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)
        self.G = group
        chs = [self.rep_in.size()] + num_layers * [ch]
        cout = self.rep_out.size()
        logging.info("Initing MLP")
        self.net = Sequential(
            *[MLPBlock(cin, cout) for cin, cout in zip(chs, chs[1:])],
            nn.Linear(chs[-1], cout),
        )

    def __call__(self, x, training=True):
        y = self.net(x)
        return y


@export
class Standardize(Module):
    """A convenience module to wrap a given module, normalize its input
    by some dataset x mean and std stats, and unnormalize its output by
    the dataset y mean and std stats.

    Args:
        model (Module): model to wrap
        ds_stats ((μx,σx,μy,σy) or (μx,σx)): tuple of the normalization stats

    Returns:
        Module: Wrapped model with input normalization (and output unnormalization)"""

    def __init__(self, model, ds_stats):
        super().__init__()
        self.model = model
        self.ds_stats = ds_stats

    def __call__(self, x, training):
        if len(self.ds_stats) == 2:
            muin, sin = self.ds_stats
            return self.model((x - muin) / sin, training=training)
        else:
            muin, sin, muout, sout = self.ds_stats
            y = sout * self.model((x - muin) / sin, training=training) + muout
            return y


# Networks for hamiltonian dynamics (need to sum for batched Hamiltonian grads)
@export
class MLPode(Module, metaclass=Named):
    def __init__(self, rep_in, rep_out, group, ch=384, num_layers=3):
        super().__init__()
        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)
        self.G = group
        chs = [self.rep_in.size()] + num_layers * [ch]
        cout = self.rep_out.size()
        logging.info("Initing MLP")
        self.net = Sequential(
            *[
                Sequential(nn.Linear(cin, cout), swish)
                for cin, cout in zip(chs, chs[1:])
            ],
            nn.Linear(chs[-1], cout),
        )

    def __call__(self, z, t):
        return self.net(z)


@export
class EMLPode(EMLP):
    """Neural ODE Equivariant MLP. Same args as EMLP."""

    # __doc__ += EMLP.__doc__.split('.')[1]
    def __init__(self, rep_in, rep_out, group, ch=384, num_layers=3):  # @
        # super().__init__()
        logging.info("Initing EMLP")
        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)
        self.G = group
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch, int):
            middle_layers = num_layers * [
                uniform_rep(ch, group)
            ]  # [uniform_rep(ch,group) for _ in range(num_layers)]
        elif isinstance(ch, Rep):
            middle_layers = num_layers * [ch(group)]
        else:
            middle_layers = [
                (c(group) if isinstance(c, Rep) else uniform_rep(c, group)) for c in ch
            ]
        # print(middle_layers[0].reps[0].G)
        # print(self.rep_in.G)
        reps = [self.rep_in] + middle_layers
        logging.info(f"Reps: {reps}")
        self.network = Sequential(
            *[EMLPBlock(rin, rout) for rin, rout in zip(reps, reps[1:])],
            Linear(reps[-1], self.rep_out),
        )

    def __call__(self, z, t):
        return self.network(z)


# Networks for hamiltonian dynamics (need to sum for batched Hamiltonian grads)
@export
class MLPH(Module, metaclass=Named):
    def __init__(self, rep_in, rep_out, group, ch=384, num_layers=3):
        super().__init__()
        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)
        self.G = group
        chs = [self.rep_in.size()] + num_layers * [ch]
        cout = self.rep_out.size()
        logging.info("Initing MLP")
        self.net = Sequential(
            *[
                Sequential(nn.Linear(cin, cout), swish)
                for cin, cout in zip(chs, chs[1:])
            ],
            nn.Linear(chs[-1], cout),
        )

    def H(self, x):  # ,training=True):
        y = self.net(x).sum()
        return y

    def __call__(self, x):
        return self.H(x)


@export
class EMLPH(EMLP):
    """Equivariant EMLP modeling a Hamiltonian for HNN. Same args as EMLP"""

    # __doc__ += EMLP.__doc__.split('.')[1]
    def H(self, x):  # ,training=True):
        y = self.network(x)
        return y.sum()

    def __call__(self, x):
        return self.H(x)


@export
@cache(maxsize=None)
def gate_indices(ch_rep: Rep) -> jnp.ndarray:
    """Indices for scalars, and also additional scalar gates
    added by gated(sumrep)"""
    channels = ch_rep.size()
    perm = ch_rep.perm
    indices = np.arange(channels)

    if not isinstance(ch_rep, SumRep):  # If just a single rep, only one scalar at end
        return (
            indices if ch_rep.is_permutation else np.ones(ch_rep.size()) * ch_rep.size()
        )

    num_nonscalars = 0
    i = 0
    for rep in ch_rep:
        if rep != Scalar and not rep.is_permutation:
            indices[perm[i : i + rep.size()]] = channels + num_nonscalars
            num_nonscalars += 1
        i += rep.size()
    return indices
