#!/usr/bin/env python3

#!/usr/bin/env python3
import jax.numpy as jnp
from jax import jit
from emlp.reps import Scalar
import numpy as np
from emlp.utils import export
from plum import dispatch
from ..group_sequences import PermutationGroupSequence, OrthogonalGroupSequence
from .linear_operators import I, LazyDirectSum, LazyKron, SlicedI, lazify, ConcatLazy
from .representation import ScalarRep, V
from .product_sum_reps import SumRep
from .utils import null_space
from functools import reduce
from collections import defaultdict


@export
class ConsistentSequence(object):
    r"""Consistent sequence of representations."""

    presentation_degree = NotImplemented
    generation_degree = NotImplemented

    def group(self, j):
        """Group at level j."""
        return self.group_sequence().group(j)

    def group_sequence(self):
        """Group sequence"""
        raise NotImplementedError

    def representation(self, j):
        """Representation at level j."""
        return V(self.group(j))

    def dimension(self, j):
        """Representation dimension at level j."""
        return self.representation(j).size()

    def up_embedding(self, j):
        r"""Embedding going from the representation at level j to level j+1."""
        raise NotImplementedError

    def composite_embedding(self, n, i):
        """Composite embedding from level n to i.

        That is, embedding(n-1) @ ... @ embedding(i).
        """

        if n <= i:
            raise ValueError(
                f"First input {n} has to be greather than the second input {i}."
            )
        elif n == i + 1:
            return self.up_embedding(i)
        else:  # n > j + 1
            return self.up_embedding(n - 1) @ self.composite_embedding(n - 1, i)

    def extendability_constraints(self, n, n0):
        """Gives constraints that extend an element at level n0 to level n."""

        constraints = []
        constraints.append(self.representation(n).constraint_matrix())
        constraints.append(self.composite_embedding(n, n0).H)
        return ConcatLazy(constraints)

    def __add__(self, other):
        """Direct sum of two representations"""
        if isinstance(other, int):
            if other == 0:
                return self
            else:
                return self + other * TrivialSequence(
                    group_sequence=self.group_sequence()
                )
        return SumSequence(self, other)

    def __radd__(self, other):
        """Direct sum of two representations"""
        if isinstance(other, int):
            if other == 0:
                return self
            else:
                return (
                    other * TrivialSequence(group_sequence=self.group_sequence()) + self
                )

        return SumSequence(self, other)

    def __mul__(self, other):
        return mul_sequences(self, other)

    def __rmul__(self, other):
        return mul_sequences(other, self)

    def __hash__(self):
        d1 = tuple(
            [
                (k, v)
                for k, v in self.__dict__.items()
                if (k not in ["_size", "is_permutation", "is_orthogonal"])
            ]
        )
        return hash((type(self), d1))

    def __rshift__(self, other):
        """Linear maps from self -> other."""
        return EquivariantOperatorSequence(self, other)

    def __lshift__(self, other):
        return EquivariantOperatorSequence(other, self)

    def __lt__(self, other):
        if isinstance(other, TrivialSequence):
            return False
        elif isinstance(self, TrivialSequence):
            return True
        try:
            if self.group_sequence() < other.group_sequence():
                return True
            elif self.group_sequence() > other.group_sequence():
                return False
        except:
            pass
        # We compare the representation at level 2 since level 1 is often trivial.
        my_rep = self.representation(2)
        other_rep = other.representation(2)
        if my_rep.size() < other_rep.size():
            return True
        elif my_rep.size() > other_rep.size():
            return False
        return hash(self) < hash(other)

    def __eq__(self, other):
        return type(self) == type(other) and type(self.group_sequence()) == type(
            other.group_sequence()
        )


@export
class GatedSequence(ConsistentSequence):
    """A Gated Sequence.

    Gated sequences contain additional trivial sequences for each irreducible
    that is not a permutation-like, i.e., representations that do not accept
    component-wise activation function.
    """

    def __init__(self, input: ConsistentSequence):
        self._original_sequence = input
        self.generation_degree = input.generation_degree
        self.presentation_degree = input.presentation_degree

        if isinstance(input, SumSequence):
            # print("It is an instance")
            self._gated_sequence = input + sum(
                [
                    TrivialSequence(input.group_sequence())
                    for sequence in input
                    if not isinstance(sequence, TrivialSequence)
                    and not sequence.is_permutation
                ]
            )
        else:
            # print("It is not an instance")
            self._gated_sequence = (
                input + TrivialSequence(input.group_sequence())
                if not input.is_permutation
                else input
            )

    def group_sequence(self):
        return self._original_sequence.group_sequence()

    # def dimension(self, j):
    #     return self._gated_sequence.size()

    def up_embedding(self, j):
        return self._gated_sequence.up_embedding(j)

    def representation(self, j):
        return self._gated_sequence.representation(j)

    def extendability_constraints(self, n, n0):
        return self._gated_sequence.extendability_constraints(n, n0)


@export
def bilinear_aux(rep_in, rep_out):
    """Outputs a function that takes as input element x in rep_in and returns a mapping rep_in -> rep_out."""
    W_rep, W_perm = (rep_in >> rep_out).canonicalize()
    inv_perm = np.argsort(W_perm)
    mat_shape = rep_out.size(), rep_in.size()
    x_rep = rep_in if isinstance(rep_in, SumRep) else SumRep(rep_in)
    W_multiplicities = W_rep.reps
    x_multiplicities = x_rep.reps
    x_multiplicities = {rep: n for rep, n in x_multiplicities.items() if rep != Scalar}
    nelems = lambda nx: min(nx, 10)
    # import pdb

    # pdb.set_trace()

    reduced_indices_dict = {
        rep: ids[np.arange(nelems(len(ids)))].reshape(-1)
        for rep, ids in x_rep.as_dict(np.arange(x_rep.size())).items()
    }
    param_dims = sum([W_multiplicities.get(rep, 0) * nelems(n) for rep, n in x_multiplicities.items()])

    @jit
    def lazy_projection(params, x):
        bshape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        bs = x.shape[0]
        i = 0
        Ws = []
        for rep, W_mult in W_multiplicities.items():
            if rep not in x_multiplicities:
                Ws.append(jnp.zeros((bs, W_mult * rep.size())))
                continue
            x_mult = x_multiplicities[rep]
            n = nelems(x_mult)
            i_end = i + W_mult * n
            bids = reduced_indices_dict[rep]
            bilinear_params = params[i:i_end].reshape(W_mult, n)  # bs,nK-> (nK,bs)
            i = i_end  # (bs,W_mult,d^r) = (W_mult,n)@(n,d^r,bs)
            bilinear_elems = bilinear_params @ x[..., bids].T.reshape(
                n, rep.size() * bs
            )
            bilinear_elems = bilinear_elems.reshape(W_mult * rep.size(), bs).T
            Ws.append(bilinear_elems)
        Ws = jnp.concatenate(Ws, axis=-1)  # concatenate over rep axis
        return Ws[..., inv_perm].reshape(
            *bshape, *mat_shape
        )  # reorder to original rank ordering

    return param_dims, lazy_projection


# --------------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------------
# The  following define sums and tensor products of consistent sequences.
# *Warning* The way we compute the presentation and generation degree only works
# for permutation groups. It is unclear how to compute these for general representations.
# TODO: Have a more robust way to compute these degrees for other representations.


class SumSequence(ConsistentSequence):
    """Sum sequence between two sequences"""

    def __init__(self, *sequences):
        """Constructs a sum sequence from a list of sequences."""
        sequences = [
            SumSequenceFromCollection({TrivialSequence: seq})
            if isinstance(seq, int)
            else seq
            for seq in sequences
        ]
        seq_counters = [
            seq.sequences if isinstance(seq, SumSequence) else {seq: 1}
            for seq in sequences
        ]
        self.sequences = self._compute_canonical(seq_counters)
        self.generation_degree = reduce(
            max, [seq.generation_degree for seq in self.sequences.keys()]
        )
        self.presentation_degree = reduce(
            max, [seq.presentation_degree for seq in self.sequences.keys()]
        )
        self._group_sequence = next(iter(self.sequences)).group_sequence()
        self.is_permutation = all([seq.is_permutation for seq in self.sequences.keys()])
        # import pdb; pdb.set_trace()
        # self._num_summands = sum([ for ])

    def _compute_canonical(self, seq_counters):
        unique_seq = sorted(
            reduce(lambda a, b: a | b, [seq.keys() for seq in seq_counters])
        )
        merged_counts = defaultdict(int)
        for seq in unique_seq:
            for cs in seq_counters:
                if seq in cs:
                    merged_counts[seq] += cs[seq]
        return merged_counts

    def num_sumands(self):
        pass

    # return len(sel)

    def __hash__(self):
        return hash(tuple(self.sequences.items()))

    # def __init__(self, first_sequence, second_sequence):
    #     """Initialize with the two summands."""
    #     self.first_sequence = first_sequence
    #     self.second_sequence = second_sequence
    #     self.presentation_degree = max(
    #         first_sequence.presentation_degree, second_sequence.presentation_degree
    #     )
    #     self.generation_degree = max(
    #         first_sequence.generation_degree, second_sequence.generation_degree
    #     )
    #     self.is_permutation = first_sequence.is_permutation and second_sequence.is_permutation
    #     self._group_sequence = first_sequence.group_sequence()

    # def representation(self, j):
    #     """Direct sum representation"""
    #     rep = self.first_sequence.representation(
    #         j
    #     ) + self.second_sequence.representation(j)
    #     rep.G = self.group_sequence().group(j)
    #     return rep

    # def up_embedding(self, j):
    # return LazyDirectSum(
    #     [self.first_sequence.up_embedding(j), self.second_sequence.up_embedding(j)]
    # )

    def representation(self, j):
        reps = [count * seq.representation(j) for seq, count in self.sequences.items()]
        return SumRep(*reps)

    def up_embedding(self, j):
        """Direct sum of the embeddings"""
        up_embeddings = [seq.up_embedding(j) for seq in self.sequences]
        multiplicities = self.sequences.values()
        return LazyDirectSum(up_embeddings, multiplicities)

    def group_sequence(self):
        """Group sequence shared by both summands"""
        return self._group_sequence

    def __repr__(self):
        return "+".join(
            f"{count if count > 1 else ''}{repr(sequence)}"
            for sequence, count, in self.sequences.items()
        )

    def __str__(self):
        return "+".join(
            f"{count if count > 1 else ''}{sequence}"
            for sequence, count, in self.sequences.items()
        )

    def __iter__(self):
        return (seq for seq, counter in self.sequences.items() for _ in range(counter))

    def __len__(self):
        return sum(multiplicity for multiplicity in self.sequences.values())

    def __eq__(self, other):
        return isinstance(other, SumSequence) and self.sequences == other.sequences


class SumSequenceFromCollection(SumSequence):
    def __init__(self, counter, perm=None):
        self.sequences = counter
        self.perm = np.arange(self.num_sumands()) if perm is None else perm
        self.sequences, self.perm = self._compute_canonical([counter], [self.perm])
        self.invperm = np.argsort(self.perm)
        self.is_permutation = all(rep.is_permutation for rep in self.sequences.keys())


class ProductSequence(ConsistentSequence):
    """Product sequence between two sequences."""

    def group_sequence(self):
        return self.first_sequence.group_sequence()

    def __init__(self, first_sequence, second_sequence):
        self.first_sequence = first_sequence
        self.second_sequence = second_sequence
        self.presentation_degree = max(
            first_sequence.presentation_degree + second_sequence.generation_degree,
            first_sequence.generation_degree + second_sequence.presentation_degree,
        )
        self.generation_degree = (
            first_sequence.generation_degree + second_sequence.generation_degree
        )
        self.is_permutation = (
            first_sequence.is_permutation and second_sequence.is_permutation
        )

    def representation(self, j):
        r"""Tensor product representation"""
        return self.first_sequence.representation(
            j
        ) * self.second_sequence.representation(j)

    def up_embedding(self, j):
        r"""Kroneker product of the embeddings"""
        return LazyKron(
            # [self.first_sequence.up_embedding(j), self.second_sequence.up_embedding(j)]
            [self.second_sequence.up_embedding(j), self.first_sequence.up_embedding(j)]
        )

    def __hash__(self):
        return hash((self.first_sequence, self.second_sequence))

    def __eq__(self, other):
        return (
            isinstance(other, ProductSequence)
            and self.first_sequence == other.first_sequence
            and self.second_sequence == other.second_sequence
        )


@dispatch
def mul_sequences(sequence_a, sequence_b: int):
    if sequence_b == 1:
        return sequence_a
    elif sequence_b == 0:
        return 0
    else:
        return SumSequence(*(sequence_b * [sequence_a]))


@dispatch
def mul_sequences(sequence_a: int, sequence_b):
    return mul_sequences(sequence_b, sequence_a)


@dispatch
def mul_sequences(sequence_a, sequence_b):
    if type(sequence_a) is TrivialSequence:
        return sequence_b
    if type(sequence_b) is TrivialSequence:
        return sequence_a
    return ProductSequence(sequence_a, sequence_b)


# --------------------------------------------------------------------------------
# Linear map sequences
# --------------------------------------------------------------------------------
# We handle operators sequences in a different fashion than general representations
# in order to have direct access to the embeddings of both input and output
# representations.


@export
class EquivariantOperatorSequence(object):
    """Sequence of equivariant linear operator spaces."""

    def __init__(self, input_representation, output_representation=None):
        """Init method."""
        self.input_representation = input_representation
        if output_representation is None:
            self.output_representation = input_representation
        else:
            self.output_representation = output_representation

    def compatibility_constraints(self, j):
        """Constraints that ensure that the basis at level j is extendable."""
        constraints = []
        pre_dgr = self.input_representation.presentation_degree
        if j < pre_dgr:
            raise ValueError(
                f"Can only extend when the level {j} is equal or larger than the presentation degree {pre_dgr}"
            )

        constraints.extend(
            [
                # Their kronecker version is backwards why?!
                LazyKron(
                    [
                        (
                            I(self.output_representation.dimension(j))
                            - self.output_representation.composite_embedding(j, k)
                            @ self.output_representation.composite_embedding(j, k).H
                        ),
                        self.input_representation.composite_embedding(j, k).H,
                    ]
                )
                for k in range(
                    1, min(self.input_representation.presentation_degree, j - 1) + 1
                )
            ]
        )
        return ConcatLazy(constraints)

    def equivariant_basis(self, level):
        return self.at_level(level).equivariant_basis()

    def composite_embedding(self, up_level, low_level):
        return LazyKron(
            [
                self.output_representation.composite_embedding(up_level, low_level),
                self.input_representation.composite_embedding(up_level, low_level),
            ]
        )

    def extendability_constraints(self, n, n0):
        constraints = []
        constraints.append(
            (
                self.input_representation.representation(n)
                >> self.output_representation.representation(n)
            ).constraint_matrix()
        )
        constraints.append(
            LazyKron(
                [
                    self.output_representation.composite_embedding(n, n0).H,
                    self.input_representation.composite_embedding(n, n0).H,
                ]
            )
        )
        return ConcatLazy(constraints)

    def at_level(self, j):
        return EquivariantOperators(
            self.input_representation.representation(j),
            self.output_representation.representation(j),
            self.compatibility_constraints(j),
        )


@export
class EquivariantOperators(object):
    def __init__(
        self,
        input_representation,
        output_representation,
        compatibility_constraints=None,
    ):
        self.input_representation = input_representation
        self.output_representation = output_representation
        self.compatibility_constraints = compatibility_constraints

    def equivariant_basis(self):
        linear_maps = self.input_representation >> self.output_representation
        basis = linear_maps.equivariant_basis()
        if self.compatibility_constraints is not None:
            coefficients = null_space(self.compatibility_constraints @ lazify(basis))
            basis = basis @ coefficients
        return basis


# --------------------------------------------------------------------------------
# Consistent sequence implementations
# --------------------------------------------------------------------------------
@export
class TrivialSequence(ConsistentSequence):
    r"""Trivial sequence, the representations at all levels are 1."""

    def __init__(self, group_sequence=None):
        self._group_sequence = group_sequence
        self.presentation_degree = 1
        self.generation_degree = 1
        self.is_permutation = True

    def group(self, j):
        """Return the group at level j."""
        if self.group_sequence is None:
            return None
        return self.group_sequence.group(j)

    def group_sequence(self):
        """Return group sequence."""
        return self._group_sequence

    def representation(self, j):
        """Return trivial representation."""
        return ScalarRep(self.group_sequence().group(j))

    def up_embedding(self, j):
        """Identity embedding."""
        return lazify(np.eye(1))

    def __eq__(self, other):
        return isinstance(other, TrivialSequence)

    def __hash__(self):
        return 0


@export
class PermutationSequence(ConsistentSequence):
    """Permutation group representation sequence."""

    def __init__(self):
        """Initialize the sequence."""
        self.presentation_degree = 2
        self.generation_degree = 2
        self._group_sequence = PermutationGroupSequence()
        self.is_permutation = True

    def group_sequence(self):
        """Return the permutation group in j elements."""
        return self._group_sequence

    def up_embedding(self, j):
        """Pad with one zero."""
        return SlicedI(j + 1, j)


@export
class OrthogonalSequence(ConsistentSequence):
    """Orthogonal group representation sequence."""

    def __init__(self):
        """Initialize the sequence."""
        self.presentation_degree = (
            1  # It is unclear whether this is the case, but it seems to be true.
        )
        self.generation_degree = 1
        self._group_sequence = OrthogonalGroupSequence()
        self.is_permutation = False

    def group_sequence(self):
        """Return the permutation group in j elements."""
        return self._group_sequence

    def up_embedding(self, j):
        """Pad with one zero."""
        return SlicedI(j + 1, j)
